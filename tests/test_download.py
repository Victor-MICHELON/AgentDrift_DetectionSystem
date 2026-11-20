"""Checksum verification of raw data (no network)."""

import hashlib

import pytest

from agentdrift.config import load_config
from agentdrift.data.download import DataIntegrityError, sha256, verify_checksums


def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_bytes(b'{"id": 1}\n' * 1000)
    assert sha256(f) == hashlib.sha256(f.read_bytes()).hexdigest()


def test_verify_accepts_intact_file(tmp_path):
    f = tmp_path / "sub" / "a.jsonl"
    f.parent.mkdir()
    f.write_text("ok\n")
    verify_checksums(tmp_path, {"sub/a.jsonl": sha256(f)})


def test_verify_rejects_modified_and_missing_files(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_text("original\n")
    expected = {"a.jsonl": sha256(f), "b.jsonl": "0" * 64}
    f.write_text("tampered\n")
    with pytest.raises(DataIntegrityError) as exc:
        verify_checksums(tmp_path, expected)
    msg = str(exc.value)
    assert "checksum mismatch: a.jsonl" in msg and "missing: b.jsonl" in msg


def test_data_config_pins_versions_and_checksums():
    cfg = load_config("data")
    assert len(cfg["agentdrift"]["commit"]) == 40
    assert len(cfg["atbench"]["revision"]) == 40
    for ds in ("agentdrift", "atbench"):
        assert not cfg[ds]["raw_dir"].startswith("/")
        assert all(len(d) == 64 for d in cfg[ds]["sha256"].values())
