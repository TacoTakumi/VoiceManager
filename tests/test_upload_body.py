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


# --- the handler reads nothing before it refuses ---------------------------------------

import asyncio
import sys
import types
from datetime import datetime

from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.session import SessionEntry, SessionRegistry

WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100
EXTRACTED = object()


class BodyCounter:
    """ASGI wrapper that counts the request-body bytes the app pulls from the connection."""

    def __init__(self, app):
        self.app = app
        self.bytes_read = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def counting_receive():
            message = await receive()
            if message["type"] == "http.request":
                self.bytes_read += len(message.get("body", b""))
            return message

        await self.app(scope, counting_receive, send)


def _app(tts_engine_type: str = "openvoice"):
    """A model-free app with a registry, a fake TTS engine and a byte counter."""
    app = create_app()
    app.state.tts_engine_type = tts_engine_type
    app.state.session_registry = SessionRegistry()
    app.state.tts_engine = types.SimpleNamespace(tone_color_converter=object())
    counter = BodyCounter(app)
    return app, counter, TestClient(counter)


def _live_session(app, owner: str = authkit.USER) -> SessionEntry:
    """Register a session owned by ``owner`` with a pipeline that takes a voice profile."""
    entry = SessionEntry(
        session_id="live-session",
        connected_at=datetime.now(),
        last_activity=datetime.now(),
        tts_pipeline=types.SimpleNamespace(target_se=None),
        owner=owner,
    )
    asyncio.run(app.state.session_registry.register(entry))
    return entry


@pytest.fixture
def fake_extractor(monkeypatch):
    """Stand in for openvoice.se_extractor so no model is imported or run."""
    calls = []

    def get_se(path, converter, vad=True):
        calls.append(path)
        return EXTRACTED, None

    se_extractor = types.SimpleNamespace(get_se=get_se)
    monkeypatch.setitem(sys.modules, "openvoice", types.SimpleNamespace(se_extractor=se_extractor))
    monkeypatch.setitem(sys.modules, "openvoice.se_extractor", se_extractor)
    return calls


def _post(client, headers, session_id="live-session", files=None):
    if files is None:
        files = {"file": ("test.wav", WAV, "audio/wav")}
    return client.post("/voice/upload", params={"session_id": session_id},
                       headers=headers, files=files)


def test_missing_token_refused_before_the_body_is_read():
    app, counter, client = _app()
    resp = _post(client, headers={})
    assert resp.status_code == 401
    assert counter.bytes_read == 0


def test_revoked_token_refused_before_the_body_is_read():
    name = "upload-body-revoked"
    secret = authkit.mint(name)
    authkit.revoke(name)
    app, counter, client = _app()
    resp = _post(client, headers=authkit.bearer_headers(secret))
    assert resp.status_code == 401
    assert counter.bytes_read == 0


@pytest.mark.parametrize("engine", ["vibevoice", "audiocpp", "voxtream"])
def test_engine_without_cloning_refused_before_the_body_is_read(engine):
    app, counter, client = _app(engine)
    _live_session(app)
    resp = _post(client, headers=authkit.auth_headers())
    assert resp.status_code == 400
    assert counter.bytes_read == 0


def test_unknown_session_refused_before_the_body_is_read():
    app, counter, client = _app()
    resp = _post(client, headers=authkit.auth_headers(), session_id="nope")
    assert resp.status_code == 404
    assert counter.bytes_read == 0


def test_form_without_file_field_is_400(fake_extractor):
    app, counter, client = _app()
    _live_session(app)
    resp = _post(client, headers=authkit.auth_headers(),
                 files={"other": ("note.txt", b"hello", "text/plain")})
    assert resp.status_code == 400
    assert fake_extractor == []


def test_empty_file_is_400(fake_extractor):
    app, counter, client = _app()
    _live_session(app)
    resp = _post(client, headers=authkit.auth_headers(),
                 files={"file": ("test.wav", b"", "audio/wav")})
    assert resp.status_code == 400
    assert "Empty file" in resp.json()["detail"]
    assert fake_extractor == []


def test_valid_upload_reaches_the_extractor_and_sets_the_profile(fake_extractor):
    app, counter, client = _app()
    entry = _live_session(app)
    resp = _post(client, headers=authkit.auth_headers())
    assert resp.status_code == 200
    assert resp.json() == {"status": "voice profile updated"}
    assert len(fake_extractor) == 1
    assert entry.tts_pipeline.target_se is EXTRACTED
    assert entry.has_voice_profile is True
    assert counter.bytes_read > 0


# --- the body size cap ------------------------------------------------------------------

import httpx

from voicemanager import config

CAP = 1024
BOUNDARY = b"cap-test-boundary"


@pytest.fixture
def small_cap(monkeypatch):
    monkeypatch.setattr(config.get_settings(), "upload_max_bytes", CAP)


def _multipart_chunks(content_chunks):
    """A multipart body with one file field, yielded piece by piece."""
    yield (b"--" + BOUNDARY + b"\r\n"
           b'Content-Disposition: form-data; name="file"; filename="test.wav"\r\n'
           b"Content-Type: audio/wav\r\n\r\n")
    yield from content_chunks
    yield b"\r\n--" + BOUNDARY + b"--\r\n"


async def _post_chunked(counter, chunks):
    """POST the chunks as a chunked body (no Content-Length) straight into the ASGI app."""
    async def body():
        for chunk in chunks:
            yield chunk

    transport = httpx.ASGITransport(app=counter)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        return await http.post(
            "/voice/upload", params={"session_id": "live-session"},
            headers={**authkit.auth_headers(),
                     "Content-Type": f"multipart/form-data; boundary={BOUNDARY.decode()}"},
            content=body(),
        )


def test_content_length_over_the_cap_is_413_before_any_read(small_cap, fake_extractor):
    app, counter, client = _app()
    _live_session(app)
    resp = _post(client, headers=authkit.auth_headers(),
                 files={"file": ("test.wav", b"\x00" * (CAP * 2), "audio/wav")})
    assert resp.status_code == 413
    assert str(CAP) in resp.json()["detail"]
    assert counter.bytes_read == 0
    assert fake_extractor == []


def test_chunked_body_past_the_cap_is_413_and_reading_stops(small_cap, fake_extractor):
    app, counter, client = _app()
    _live_session(app)
    chunk = 256
    chunks = list(_multipart_chunks([b"\x00" * chunk] * 16))  # 4 KiB of file content
    resp = asyncio.run(_post_chunked(counter, chunks))
    assert resp.status_code == 413
    assert CAP < counter.bytes_read <= CAP + chunk
    assert fake_extractor == []


def test_chunked_body_under_the_cap_is_accepted(small_cap, fake_extractor):
    app, counter, client = _app()
    entry = _live_session(app)
    chunks = list(_multipart_chunks([WAV[:50], WAV[50:]]))
    resp = asyncio.run(_post_chunked(counter, chunks))
    assert resp.status_code == 200
    assert len(fake_extractor) == 1
    assert entry.tts_pipeline.target_se is EXTRACTED


# --- a refused upload leaves nothing behind -------------------------------------------

import gc
import os


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_refused_chunked_upload_frees_its_spooled_file_without_the_gc(monkeypatch, fake_extractor):
    # Above 1 MiB the multipart parser spools the file part to disk; the 413 raised
    # mid-parse must not leave that file open until the cyclic GC runs.
    cap = 2 * 1024 * 1024
    monkeypatch.setattr(config.get_settings(), "upload_max_bytes", cap)
    app, counter, client = _app()
    _live_session(app)
    chunks = list(_multipart_chunks([b"\x00" * (256 * 1024)] * 12))  # 3 MiB of file content
    gc.disable()
    try:
        gc.collect()
        before = _open_fds()
        for _ in range(5):
            resp = asyncio.run(_post_chunked(counter, chunks))
            assert resp.status_code == 413
        assert _open_fds() == before
    finally:
        gc.enable()
    assert fake_extractor == []
