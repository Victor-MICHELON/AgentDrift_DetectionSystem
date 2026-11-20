"""Download raw datasets at a pinned version and verify their SHA-256 checksums.

Usage:
    python -m agentdrift.data.download               # AgentDrift: fetch if missing, then verify
    python -m agentdrift.data.download --atbench     # also ATBench (external test set)
    python -m agentdrift.data.download --verify-only # never touch the network, only check

Idempotent: a dataset already present at the pinned version with matching checksums is left as is.
Any mismatch (wrong commit, modified tracked file, wrong checksum) raises DataIntegrityError.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

from agentdrift.config import ROOT, load_config

CHUNK = 1 << 20


class DataIntegrityError(RuntimeError):
    """Raw data is missing, at the wrong version, or does not match its checksum."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums(base: Path, expected: dict[str, str]) -> None:
    """Check every `relative/path: sha256` of `expected` under `base`; report all failures at once."""
    errors = []
    for rel, digest in expected.items():
        path = base / rel
        if not path.is_file():
            errors.append(f"missing: {rel}")
        elif (actual := sha256(path)) != digest:
            errors.append(f"checksum mismatch: {rel} (expected {digest[:12]}…, got {actual[:12]}…)")
    if errors:
        raise DataIntegrityError(f"{base}:\n  " + "\n  ".join(errors))


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise DataIntegrityError(f"git {' '.join(args)} failed in {repo}: {out.stderr.strip()}")
    return out.stdout.strip()


def _clone_at_commit(source: str, commit: str, dest: Path) -> None:
    """Shallow fetch of a single commit (GitHub allows fetching by SHA)."""
    dest.mkdir(parents=True)
    _git(dest, "init", "--quiet")
    _git(dest, "remote", "add", "origin", source)
    _git(dest, "fetch", "--quiet", "--depth", "1", "origin", commit)
    _git(dest, "checkout", "--quiet", "--detach", "FETCH_HEAD")


def fetch_agentdrift(cfg: dict, verify_only: bool = False) -> Path:
    repo = ROOT / cfg["raw_dir"]
    commit = cfg["commit"]
    if not (repo / ".git").is_dir():
        if verify_only:
            raise DataIntegrityError(f"{repo} not found; run without --verify-only to download it")
        if repo.exists() and any(p.name != ".gitkeep" for p in repo.iterdir()):
            raise DataIntegrityError(f"{repo} exists but is not a git clone; remove it first")
        print(f"Fetching AgentDrift @ {commit[:7]} into {cfg['raw_dir']}")
        _clone_at_commit(cfg["source"], commit, repo)

    head = _git(repo, "rev-parse", "HEAD")
    if head != commit:
        raise DataIntegrityError(f"{repo} is at {head[:7]}, expected {commit[:7]}")
    if dirty := _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise DataIntegrityError(f"tracked files modified in {repo}:\n{dirty}")

    verify_checksums(repo, cfg["sha256"])
    print(f"AgentDrift OK: commit {commit[:7]}, {len(cfg['sha256'])} checksums verified")
    return repo


def fetch_atbench(cfg: dict, verify_only: bool = False) -> Path:
    dest = ROOT / cfg["raw_dir"]
    for rel, digest in cfg["sha256"].items():
        path = dest / rel
        if path.is_file() and sha256(path) == digest:
            continue
        if verify_only:
            break  # verify_checksums below reports the problem
        url = f"{cfg['source']}/resolve/{cfg['revision']}/{rel}"
        print(f"Downloading {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
            while chunk := resp.read(CHUNK):
                f.write(chunk)
        if (actual := sha256(tmp)) != digest:
            tmp.unlink()
            raise DataIntegrityError(f"ATBench {rel}: checksum mismatch after download ({actual[:12]}…)")
        tmp.replace(path)

    verify_checksums(dest, cfg["sha256"])
    print(f"ATBench OK: revision {cfg['revision'][:7]}, {len(cfg['sha256'])} checksum(s) verified")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atbench", action="store_true", help="also fetch the ATBench external test set")
    parser.add_argument("--verify-only", action="store_true", help="check local files, no download")
    args = parser.parse_args(argv)

    cfg = load_config("data")
    try:
        fetch_agentdrift(cfg["agentdrift"], args.verify_only)
        if args.atbench:
            fetch_atbench(cfg["atbench"], args.verify_only)
    except DataIntegrityError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
