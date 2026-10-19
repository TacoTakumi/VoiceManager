"""Structural guard: automated tests use bearer tokens, never the removed shared key.

Scans the collected test files (every tests/test_*.py except the manual
scripts listed in conftest's collect_ignore) and fails if any of them names
the legacy key env var or query parameter. The strings come from the authkit
constants so this file never contains the literals itself.
"""
from pathlib import Path

import authkit
import conftest

TESTS_DIR = Path(__file__).resolve().parent
NEEDLES = (authkit.LEGACY_KEY_ENV, authkit.LEGACY_KEY_PARAM)


def collected_test_files(tests_dir: Path = TESTS_DIR) -> list[Path]:
    ignored = set(conftest.collect_ignore)
    return sorted(p for p in tests_dir.glob("test_*.py") if p.name not in ignored)


def offenders(files: list[Path]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8")
        hits = [needle for needle in NEEDLES if needle in text]
        if hits:
            found[path.name] = hits
    return found


def test_scan_covers_a_reasonable_number_of_files():
    files = collected_test_files()
    assert len(files) > 20, f"scan found only {len(files)} test files"
    assert Path(__file__).resolve() in files  # the scan includes itself
    assert not {p.name for p in files} & set(conftest.collect_ignore)


def test_scan_detects_forbidden_strings(tmp_path):
    clean = tmp_path / "test_clean.py"
    clean.write_text("token = 'abc'\n", encoding="utf-8")
    dirty = tmp_path / "test_dirty.py"
    dirty.write_text(f"x = '{authkit.LEGACY_KEY_PARAM}'\n", encoding="utf-8")
    assert offenders([clean, dirty]) == {"test_dirty.py": [authkit.LEGACY_KEY_PARAM]}


def test_no_automated_test_uses_the_legacy_key():
    found = offenders(collected_test_files())
    assert not found, "legacy shared-key strings found in: " + ", ".join(
        f"{name} ({', '.join(hits)})" for name, hits in sorted(found.items())
    )
