"""Registry of raw data sources: download and integrity checks.

Sources are declared in ``configs/sources.yaml``. Their SHA-256 checksums live in
``configs/sources.lock.yaml``, which is written the first time a file is fetched and checked on
every later run, so a silently changed upstream file is detected instead of used.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import yaml

from asd_gene_predict.paths import CONFIGS, RAW

log = logging.getLogger(__name__)

SOURCES_FILE = CONFIGS / "sources.yaml"
LOCK_FILE = CONFIGS / "sources.lock.yaml"
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str
    description: str = ""
    version: str = ""
    stage: str = ""
    manual: bool = False
    instructions: str = ""

    def path(self, raw_dir: Path = RAW) -> Path:
        return raw_dir / self.filename


class Status(StrEnum):
    DOWNLOADED = "downloaded"
    OK = "ok"  # present and matches the lock
    LOCKED = "locked"  # present, no previous lock entry, checksum recorded now
    MISSING = "missing"  # manual source not yet placed in data/raw
    MISMATCH = "mismatch"  # present but checksum differs from the lock


@dataclass(frozen=True)
class FetchResult:
    source: Source
    status: Status
    sha256: str | None = None
    message: str = ""


def load_sources(path: Path = SOURCES_FILE) -> dict[str, Source]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)["sources"]
    return {
        name: Source(name=name, **{k: str(v) if k == "version" else v for k, v in spec.items()})
        for name, spec in raw.items()
    }


def load_lock(path: Path = LOCK_FILE) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_lock(lock: dict[str, dict], path: Path = LOCK_FILE) -> None:
    header = "# Gerado pelo `asd fetch`. Não editar à mão; usar `asd fetch --update-lock`.\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(dict(sorted(lock.items())), f, sort_keys=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, timeout: float = 60) -> None:
    """Stream ``url`` to ``dest`` via a ``.part`` file, so an interrupted download never
    leaves a truncated file with the final name."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "asd-gene-predict"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out, _CHUNK)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def select_sources(
    sources: dict[str, Source], names: Iterable[str] = (), stage: str | None = None
) -> list[Source]:
    names = list(names)
    unknown = sorted(set(names) - sources.keys())
    if unknown:
        raise KeyError(f"Fontes desconhecidas: {unknown}. Disponíveis: {sorted(sources)}")
    selected = [sources[n] for n in names] if names else list(sources.values())
    if stage:
        selected = [s for s in selected if s.stage == stage]
    return selected


def fetch(
    sources: Iterable[Source],
    raw_dir: Path = RAW,
    lock_path: Path = LOCK_FILE,
    force: bool = False,
    update_lock: bool = False,
) -> list[FetchResult]:
    """Download (if needed) and verify each source against the lock file.

    - Automatic sources are downloaded when missing, or always with ``force``.
    - Manual sources are never downloaded; a missing file is reported with instructions.
    - A file whose checksum differs from the lock is reported as ``MISMATCH`` and the lock
      is left untouched, unless ``update_lock`` is set.
    """
    lock = load_lock(lock_path)
    lock_changed = False
    results: list[FetchResult] = []

    for src in sources:
        dest = src.path(raw_dir)
        downloaded = False

        if src.manual:
            if not dest.exists():
                results.append(FetchResult(src, Status.MISSING, message=src.instructions.strip()))
                continue
        elif force or not dest.exists():
            log.info("A descarregar %s de %s", src.name, src.url)
            download(src.url, dest)
            downloaded = True

        digest = sha256_file(dest)
        previous = lock.get(src.name, {}).get("sha256")

        if previous and previous != digest and not update_lock:
            results.append(
                FetchResult(
                    src,
                    Status.MISMATCH,
                    digest,
                    f"SHA-256 esperado {previous[:12]}…, obtido {digest[:12]}…. "
                    "Se a mudança for intencional, correr com --update-lock.",
                )
            )
            continue

        if previous != digest:
            lock[src.name] = {
                "filename": src.filename,
                "version": src.version,
                "sha256": digest,
                "locked_at": datetime.now(UTC).strftime("%Y-%m-%d"),
            }
            lock_changed = True
        if downloaded:
            status = Status.DOWNLOADED
        else:
            status = Status.OK if previous == digest else Status.LOCKED
        results.append(FetchResult(src, status, digest))

    if lock_changed:
        save_lock(lock, lock_path)
    return results
