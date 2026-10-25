"""Local Forge script presence and AI hints, never a rules-correctness certificate."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from typing import Any

from .cards import name_key
from .forge import forge_name
from .oracle import OracleStore


class ForgeSupport:
    def __init__(self, cards_dir: str | Path, *, known_issues: dict[str, str] | None = None):
        self.cards_dir = Path(cards_dir).resolve()
        if not self.cards_dir.is_dir():
            raise ValueError("Forge cardsfolder directory does not exist")
        self.issues = {name_key(n): reason for n, reason in (known_issues or {}).items()}
        self.scripts: dict[str, dict[str, Any]] = {}
        digest = hashlib.sha256()
        for archive in sorted(self.cards_dir.glob("*.zip")):
            with zipfile.ZipFile(archive) as z:
                for info in sorted(z.infolist(), key=lambda x: x.filename):
                    if info.filename.endswith(".txt") and not info.is_dir():
                        self._add(z.read(info), str(archive) + "!" + info.filename, digest)
        # Loose files override archived scripts, as Forge updates/custom scripts can.
        for path in sorted(self.cards_dir.rglob("*.txt")):
            self._add(path.read_bytes(), str(path), digest)
        if not self.scripts:
            raise ValueError("No Forge card scripts found")
        self.fingerprint = digest.hexdigest()

    def _add(self, raw: bytes, source: str, digest: Any) -> None:
        text = raw.decode("utf-8-sig")
        names = re.findall(r"^Name:\s*(.+?)\s*$", text, re.M)
        if not names:
            return
        sha = hashlib.sha256(raw).hexdigest()
        digest.update(("\0".join(names) + "\0" + sha + "\n").encode())
        record = {
            "forge_name": names[0],
            "names": names,
            "script": source,
            "sha256": sha,
            "ai_hints": [
                line.strip() for line in re.findall(r"^(?:AI:.*|SVar:RemAIDeck:.*)$", text, re.M)
            ],
        }
        for n in names:
            self.scripts[name_key(n)] = record
        self.scripts[name_key(" // ".join(names))] = record

    def check(self, deck: dict[str, Any], store: OracleStore) -> dict[str, Any]:
        if (
            not isinstance(deck, dict)
            or not isinstance(deck.get("commander"), str)
            or not isinstance(deck.get("cards"), dict)
        ):
            raise ValueError("Forge support checks require one commander and a cards mapping")
        records, missing, issues, warnings = [], [], [], []
        for name in dict.fromkeys([deck["commander"], *deck["cards"]]):
            card = store.get_card(name)
            if card is None:
                missing.append({"card": name, "reason": "No exact Oracle record"})
                continue
            script = self.scripts.get(name_key(forge_name(card)))
            if script is None:
                missing.append({"card": card["name"], "reason": "No local Forge script"})
                continue
            required_faces = {name_key(f["name"]) for f in card.get("card_faces", [])}
            if required_faces - {name_key(n) for n in script["names"]}:
                missing.append(
                    {"card": card["name"], "reason": "Script lacks an expected card face"}
                )
            entry = {"card": card["name"], **script}
            records.append(entry)
            issue = self.issues.get(name_key(card["name"]))
            if issue:
                issues.append({"card": card["name"], "reason": issue})
            if script["ai_hints"]:
                warnings.append(
                    {
                        "card": card["name"],
                        "hints": script["ai_hints"],
                        "reason": "Forge AI/deck-generation hint, not proof of a rules bug.",
                    }
                )
        return {
            "supported": not missing and not issues,
            "missing": missing,
            "known_issues": issues,
            "ai_warnings": warnings,
            "cards": records,
            "scripts_fingerprint": self.fingerprint,
            "scope": "Presence and face checks only; scripts and Forge AI can still misbehave.",
        }
