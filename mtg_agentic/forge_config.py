"""Pinned Forge installations and reproducible three-opponent pods."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .forge import ForgeError
from .schema import load_json


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dck(path: str | Path) -> dict[str, Any]:
    """Read only single-commander Main/Commander sections, including Forge print suffixes."""
    section = ""
    main: Counter[str] = Counter()
    commanders: Counter[str] = Counter()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].casefold()
            continue
        if section not in {"main", "commander"}:
            continue
        match = re.fullmatch(r"(\d+)\s+(.+)", line)
        if not match or int(match[1]) <= 0:
            raise ForgeError("Unsupported Forge deck entry")
        # Optional '+' foil marker and |SET|art fields are not part of the name.
        name = match[2].split("|", 1)[0].rstrip("+").strip()
        (commanders if section == "commander" else main)[name] += int(match[1])
    if sum(commanders.values()) != 1 or len(commanders) != 1:
        raise ForgeError(
            "Exactly one Commander section card is required; partner/background unsupported"
        )
    return {"commander": next(iter(commanders)), "cards": dict(main), "bracket": 4}


def _properties(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="iso-8859-1").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "!")) or "=" not in line:
            continue
        key, value = line.split("=", 1)

        # Java .properties escaping, including escaped backslashes in Windows paths.
        def unescape(match):
            escaped = match[1]
            if escaped.startswith("u") and len(escaped) == 5:
                return chr(int(escaped[1:], 16))
            return {"t": "\t", "n": "\n", "r": "\r", "f": "\f"}.get(escaped, escaped)

        result[key.strip()] = re.sub(r"\\(u[0-9a-fA-F]{4}|.)", unescape, value).strip()
    return result


@dataclass(frozen=True)
class ForgeInstallation:
    home: Path
    jar: Path
    java: str
    decks_dir: Path
    jar_sha256: str
    release: str

    @classmethod
    def inspect(
        cls, home: str | Path, *, decks_dir: str | Path | None = None, java: str | None = None
    ) -> ForgeInstallation:
        home = Path(home).resolve()
        jars = list(home.glob("forge-gui-desktop-*-jar-with-dependencies.jar"))
        if len(jars) != 1:
            raise ForgeError("Forge home must contain exactly one desktop executable JAR")
        java_path = shutil.which(java or "java")
        if not java_path:
            raise ForgeError("Java is not installed or the executable was not found")
        props = _properties(home / "forge.profile.properties")
        if decks_dir is None:
            if os.name == "nt":
                default = Path(os.getenv("APPDATA", str(Path.home() / ".forge"))) / "Forge"
            elif sys.platform == "darwin":
                default = Path.home() / "Library/Application Support/Forge"
            else:
                default = Path.home() / ".forge"
            user = Path(props.get("userDir") or default)
            if not user.is_absolute():
                user = home / user
            decks = Path(props.get("decksDir") or user / "decks")
            if not decks.is_absolute():
                decks = home / decks
            decks_dir = decks / "commander"
        return cls(
            home,
            jars[0],
            str(Path(java_path).resolve()),
            Path(decks_dir).resolve(),
            file_hash(jars[0]),
            jars[0]
            .name.removeprefix("forge-gui-desktop-")
            .removesuffix("-jar-with-dependencies.jar"),
        )

    def command(self) -> list[str]:
        # forge.cmd in 2.0.15 does not forward command-line arguments.
        return [
            self.java,
            "-Xmx2g",
            "-Djava.awt.headless=true",
            "-Dfile.encoding=UTF-8",
            "-jar",
            str(self.jar),
            "sim",
        ]

    def provenance(self) -> dict[str, str]:
        return {
            "release": self.release,
            "jar_sha256": self.jar_sha256,
            "home": str(self.home),
            "jar": str(self.jar),
        }


@dataclass(frozen=True)
class OpponentPod:
    name: str
    opponents: tuple[Path, ...]
    hashes: tuple[str, ...]
    expected_release: str

    @classmethod
    def load(cls, path: str | Path, installation: ForgeInstallation) -> OpponentPod:
        data = load_json(Path(path).read_text(encoding="utf-8-sig"))
        if (
            not isinstance(data, dict)
            or data.get("version") != 1
            or not isinstance(data.get("name"), str)
            or not data["name"].strip()
            or not isinstance(data.get("opponents"), list)
            or len(data["opponents"]) != 3
        ):
            raise ValueError("Pod config requires version 1, a name and exactly three opponents")
        expected_release = data.get("forge_release", "")
        if expected_release != installation.release:
            raise ForgeError("Pod Forge release does not match the installed release")
        opponents, hashes = [], []
        for entry in data["opponents"]:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("path"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", entry.get("sha256", ""))
            ):
                raise ValueError("Every opponent needs a path and pinned SHA-256")
            candidate = Path(entry["path"])
            if not candidate.is_absolute():
                candidate = installation.home / candidate
            candidate = candidate.resolve()
            if candidate.suffix.lower() != ".dck":
                raise ValueError("Opponents must be .dck files")
            digest = file_hash(candidate)
            if digest != entry["sha256"]:
                raise ForgeError("Opponent changed since this pod was pinned: " + str(candidate))
            read_dck(candidate)
            opponents.append(candidate)
            hashes.append(digest)
        return cls(data["name"], tuple(opponents), tuple(hashes), expected_release)

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "opponents": [
                {"path": str(p), "sha256": digest} for p, digest in zip(self.opponents, self.hashes)
            ],
            "forge_release": self.expected_release,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "name": self.name,
                    "opponents": self.hashes,
                    "forge_release": self.expected_release,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
