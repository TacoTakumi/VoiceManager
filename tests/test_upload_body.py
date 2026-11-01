"""POST /voice/upload refuses before it reads the body, and caps the body size.

Model-free and server-free: the app is driven through an ASGI wrapper that counts
the body bytes the handler pulls from the connection, so each test can assert that
a refusal read nothing. The tone-colour extractor is faked.
"""
import pytest

import authkit  # test auth env; must be imported before voicemanager

from voicemanager.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("VM_UPLOAD_MAX_BYTES", raising=False)


def test_upload_max_bytes_default_is_20_mib():
    assert Settings(_env_file=None).upload_max_bytes == 20 * 1024 * 1024


def test_upload_max_bytes_env_override(monkeypatch):
    monkeypatch.setenv("VM_UPLOAD_MAX_BYTES", "1024")
    assert Settings(_env_file=None).upload_max_bytes == 1024


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_upload_max_bytes_non_positive_rejected(monkeypatch, bad):
    from pydantic import ValidationError
    monkeypatch.setenv("VM_UPLOAD_MAX_BYTES", bad)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_upload_max_bytes_listed_in_env_example():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    assert "# VM_UPLOAD_MAX_BYTES=20971520" in text
