"""Opt-in local Hugging Face embeddings: Google EmbeddingGemma 2 for rules retrieval.

Loads only the text tower (270M of 740M parameters) and replicates the model's
SentenceTransformers head: mean pooling over every token, then L2 normalization. Needs
`transformers>=5.21` and `torch`; nothing loads unless this backend is selected.
"""

from __future__ import annotations

import logging
import time

EMBEDDINGGEMMA_MODEL = "google/embeddinggemma-2"
EMBEDDINGGEMMA_REVISION = "26711154b641697c443124e316f3d21587f9ff5d"
# Model-card task prefixes. Interaction checks ask questions, so queries use question answering.
QUERY_PREFIX = "task: question answering | query: "
DOCUMENT_TEMPLATE = "title: {title} | text: {text}"
# Changing the input format changes every vector, so it is part of the index identity.
FORMAT_VERSION = "qa-titled-v1"
DIMENSIONS = (128, 256, 512, 768)
MAX_TOKENS = 8192

log = logging.getLogger(__name__)


def format_inputs(texts: list[str], *, titles: list[str] | None = None, query: bool = False) -> list[str]:
    if titles is not None and len(titles) != len(texts):
        raise ValueError("Embedding titles must match the texts")
    if query:
        return [QUERY_PREFIX + text for text in texts]
    return [
        DOCUMENT_TEMPLATE.format(title=(titles[i] if titles else "") or "none", text=text)
        for i, text in enumerate(texts)
    ]


class EmbeddingGemmaEmbedder:
    def __init__(
        self,
        model: str = EMBEDDINGGEMMA_MODEL,
        *,
        revision: str | None = EMBEDDINGGEMMA_REVISION,
        device: str | None = None,
        dimension: int = 768,
        batch_size: int = 16,
    ):
        if dimension not in DIMENSIONS:
            raise ValueError(f"EmbeddingGemma 2 dimension must be one of {DIMENSIONS}")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Embedding batch size must be a positive integer")
        try:
            import torch
            import transformers
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError:
            raise ValueError(
                'EmbeddingGemma 2 needs: pip install "transformers>=5.21" torch'
            ) from None
        if not hasattr(transformers, "EmbeddingGemma2Model"):
            raise ValueError(
                f"EmbeddingGemma 2 needs transformers>=5.21, found {transformers.__version__}"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # The model card forbids float16: its activations overflow into NaN or silent garbage.
        bf16 = self.device.startswith("cuda") and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float32
        self.model, self.dimension, self.batch_size, self._torch = model, dimension, batch_size, torch
        self.identity = f"hf:{model}@{revision or 'main'}:{dimension}:{FORMAT_VERSION}"
        start = time.monotonic()
        config = AutoConfig.from_pretrained(
            model, revision=revision, vision_config=None, audio_config=None
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self._model = (
            AutoModel.from_pretrained(model, revision=revision, config=config, dtype=dtype)
            .to(self.device)
            .eval()
        )
        log.info("embeddings: loaded %s on %s in %.1fs", model, self.device, time.monotonic() - start)

    def embed(
        self, texts: list[str], *, titles: list[str] | None = None, query: bool = False
    ) -> list[list[float]]:
        torch = self._torch
        inputs = format_inputs(texts, titles=titles, query=query)
        vectors: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(inputs), self.batch_size):
                batch = self._tokenizer(
                    inputs[start : start + self.batch_size], padding=True, return_tensors="pt"
                )
                if batch["input_ids"].shape[1] > MAX_TOKENS:
                    raise ValueError(f"Embedding input exceeds {MAX_TOKENS} tokens")
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                hidden = self._model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
                weights = mask.unsqueeze(-1).float()
                pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled[:, : self.dimension], dim=-1)
                vectors.extend(pooled.cpu().tolist())
        return vectors
