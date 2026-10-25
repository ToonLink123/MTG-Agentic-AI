"""Read-only Git identity, including a digest of uncommitted source changes."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def repository_state(root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root or Path(__file__).resolve().parent.parent).resolve()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, check=True, timeout=15
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        dirty = bool(git("status", "--porcelain"))
        digest = hashlib.sha256(git("diff", "--no-ext-diff", "--binary", "HEAD"))
        for name in sorted(git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")):
            if name:
                path = root / name.decode("utf-8")
                if path.is_file() and not path.is_symlink():
                    digest.update(name + b"\0")
                    file_digest = hashlib.sha256()
                    with path.open("rb") as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            file_digest.update(chunk)
                    digest.update(file_digest.digest())
        return {"commit": commit, "dirty": dirty, "working_tree_sha256": digest.hexdigest()}
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return {"commit": None, "dirty": None, "working_tree_sha256": None}
