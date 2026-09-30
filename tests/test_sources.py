from pathlib import Path

import pytest
import yaml

from asd_gene_predict.data.sources import (
    Source,
    Status,
    fetch,
    load_lock,
    load_sources,
    select_sources,
    sha256_file,
)


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """A local file served through a file:// URL, standing in for an upstream download."""
    f = tmp_path / "upstream" / "genes.tsv"
    f.parent.mkdir()
    f.write_text("gene\nSHANK3\n")
    return f


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "raw", tmp_path / "sources.lock.yaml"


def test_repo_registry_is_valid():
    sources = load_sources()
    assert {"hgnc", "mane", "string_aliases", "sfari", "krishnan_2016"} <= sources.keys()
    for s in sources.values():
        assert s.filename and s.url and s.stage
        assert "/" not in s.filename
    assert all(s.instructions for s in sources.values() if s.manual)


def test_select_by_stage_and_unknown_name():
    sources = load_sources()
    assert {s.name for s in select_sources(sources, stage="gene_map")} == {
        "hgnc",
        "mane",
        "string_aliases",
    }
    with pytest.raises(KeyError):
        select_sources(sources, ["nope"])


def test_download_then_ok(remote, dirs):
    raw, lock_path = dirs
    src = Source(name="demo", url=remote.as_uri(), filename="genes.tsv", version="1")

    [first] = fetch([src], raw_dir=raw, lock_path=lock_path)
    assert first.status is Status.DOWNLOADED
    assert (raw / "genes.tsv").read_text() == remote.read_text()
    assert not list(raw.glob("*.part"))
    assert load_lock(lock_path)["demo"]["sha256"] == sha256_file(remote)

    [second] = fetch([src], raw_dir=raw, lock_path=lock_path)
    assert second.status is Status.OK


def test_changed_upstream_is_detected(remote, dirs):
    raw, lock_path = dirs
    src = Source(name="demo", url=remote.as_uri(), filename="genes.tsv")
    fetch([src], raw_dir=raw, lock_path=lock_path)
    locked = load_lock(lock_path)["demo"]["sha256"]

    remote.write_text("gene\nCHD8\n")
    [res] = fetch([src], raw_dir=raw, lock_path=lock_path, force=True)
    assert res.status is Status.MISMATCH
    assert load_lock(lock_path)["demo"]["sha256"] == locked  # lock untouched

    [res] = fetch([src], raw_dir=raw, lock_path=lock_path, update_lock=True)
    assert res.status is Status.LOCKED
    assert load_lock(lock_path)["demo"]["sha256"] == sha256_file(remote)


def test_manual_source(dirs):
    raw, lock_path = dirs
    src = Source(
        name="sfari",
        url="https://example.org",
        filename="sfari.csv",
        manual=True,
        instructions="Pôr o CSV em data/raw.",
    )

    [res] = fetch([src], raw_dir=raw, lock_path=lock_path)
    assert res.status is Status.MISSING and "data/raw" in res.message
    assert not lock_path.exists()

    raw.mkdir(parents=True)
    (raw / "sfari.csv").write_text("gene-symbol\nSHANK3\n")
    [res] = fetch([src], raw_dir=raw, lock_path=lock_path)
    assert res.status is Status.LOCKED
    assert yaml.safe_load(lock_path.read_text())["sfari"]["filename"] == "sfari.csv"
