"""Shipped files carry no planning-record IDs and no pointers to private planning docs.

Requirement, task, decision and finding IDs, and the planning documents they point
at, belong to the private development record and mean nothing to a reader of the
published code. This scans every tracked file outside that private record.

Model-free and server-free. It needs a git checkout to list the tracked files and
skips without one. The samples below are assembled at runtime so this file does
not match its own scan.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Paths that stay private (not published with the code).
PRIVATE_PREFIXES = (
    "CLAUDE.md",
    "AGENTS.md",
    ".specflo/",
    ".claude/",
    "docs/projects/",
    "docs/superpowers/",
    "docs/agenda-",
    "docs/backlog.md",
    "docs/release-",
)
SELF = "tests/test_no_trace_ids.py"

_PREFIXES = "|".join(["REQ", "T", "D", "F", "M", "SESS", "PROT", "FU", "INFR", "TTS", "STT"])
# A standalone ID token, or one joined by pre-/post-/non-; not part of a longer
# name such as Breeze-TTS-2.
ID_RE = re.compile(
    rf"(?:(?<![\w-])|(?<=\bpre-)|(?<=\bpost-)|(?<=\bnon-))"
    rf"(?:{_PREFIXES})-\d{{1,3}}[a-z]?(?![\w-])")
POINTER_RE = re.compile(
    "|".join([
        r"review-findings",
        r"t14-findings",
        r"spike-record",
        r"\bspike\.md\b",
        r"docs/" + r"projects/",
        r"(?<![\w-])\.planning\b",
        r"\bPitfall \d",
        r"\bPhase \d",
        r"\breview-\d+ should-fix\b",
    ])
)

AREAS = ("src", "tests", "sidecar", "scripts", "other")


def find_trace_refs(text: str) -> list[str]:
    """Every ID or private-doc pointer in text, in order."""
    found = [(m.start(), m.group()) for m in ID_RE.finditer(text)]
    found += [(m.start(), m.group()) for m in POINTER_RE.finditer(text)]
    return [s for _, s in sorted(found)]


def _area(path: str) -> str:
    top = path.split("/", 1)[0]
    return top if top in AREAS else "other"


def _shipped_files() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout to list the tracked files")
    return [p for p in out.splitlines()
            if p != SELF and not p.startswith(PRIVATE_PREFIXES)]


def _offenders(area: str) -> list[str]:
    hits = []
    for path in _shipped_files():
        if _area(path) != area:
            continue
        try:
            text = (REPO_ROOT / path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue  # binary assets (WAVs, images) or a file deleted in the work tree
        for lineno, line in enumerate(text.splitlines(), 1):
            for ref in find_trace_refs(line):
                hits.append(f"{path}:{lineno}: {ref}")
    return hits


@pytest.mark.parametrize("sample", [
    "REQ" + "-07",
    "(T" + "-08, D" + "-10)",
    "SESS" + "-03",
    "TTS" + "-04",
    "M" + "-02 exit",
    "the pre-T" + "-24 path",
    "Pit" + "fall 3",
    "Pha" + "se 7 live verification",
    "see review" + "-findings.md",
])
def test_matcher_flags_trace_refs(sample):
    assert find_trace_refs(sample)


@pytest.mark.parametrize("sample", [
    "Breeze-TTS-2",
    "SHA-256",
    "UTF-8",
    "qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf",
    "2026-08-14",
    "a T-shaped layout",
])
def test_matcher_ignores_real_names(sample):
    assert find_trace_refs(sample) == []


@pytest.mark.parametrize("area", AREAS)
def test_no_trace_refs_in(area):
    hits = _offenders(area)
    assert not hits, f"{len(hits)} trace refs in {area}:\n" + "\n".join(hits[:200])
