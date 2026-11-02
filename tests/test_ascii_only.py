"""Every Python file in the shipped package is pure ASCII.

The repo style is ASCII only in shipped files: em dashes become ' - ', curly quotes
become straight quotes, and a regex that needs a non-ASCII character writes it as a
\\uXXXX escape. Reviews kept finding em dashes creeping back, so this test pins it.

Model-free and server-free: it reads the source tree and nothing else.
"""
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "voicemanager"

PY_FILES = sorted(PACKAGE_ROOT.rglob("*.py"))


def non_ascii_lines(data: bytes) -> list[tuple[int, str]]:
    """Return (1-based line number, offending line) for every line with a byte > 0x7F."""
    bad = []
    for number, line in enumerate(data.splitlines(), start=1):
        if any(byte > 0x7F for byte in line):
            bad.append((number, line.decode("utf-8", errors="replace")))
    return bad


def test_package_has_python_files():
    assert PY_FILES, f"no Python files found under {PACKAGE_ROOT}"


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(PACKAGE_ROOT)))
def test_shipped_python_is_ascii(path: Path):
    bad = non_ascii_lines(path.read_bytes())
    assert not bad, "non-ASCII bytes in a shipped file:\n" + "\n".join(
        f"  {path.relative_to(PACKAGE_ROOT)}:{number}: {line.strip()}" for number, line in bad
    )


def test_detector_flags_an_em_dash():
    assert non_ascii_lines("plain\nx — y\n".encode("utf-8")) == [(2, "x — y")]
    assert non_ascii_lines(b"only ascii\n") == []
