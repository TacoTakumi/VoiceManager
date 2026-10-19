"""Integration tests for session registry wiring in HTTP endpoints.

Tests use FastAPI TestClient which skips lifespan, so session registry and models
are manually attached to app.state for each test.
"""
import pytest
from fastapi.testclient import TestClient

import authkit

from voicemanager.app import create_app
from voicemanager.session import SessionRegistry, SessionEntry
from datetime import datetime, timezone


@pytest.fixture
def app_with_registry():
    """Create app with a manually-attached session registry (lifespan skipped in TestClient)."""
    app = create_app()
    registry = SessionRegistry()
    app.state.session_registry = registry
    return app, registry


def test_info_endpoint_includes_sessions_key(app_with_registry):
    """/info must include sessions list."""
    app, registry = app_with_registry
    client = TestClient(app)
    resp = client.get("/info", headers=authkit.auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert data["sessions"] == []


def test_info_endpoint_shows_registered_sessions(app_with_registry):
    """/info sessions list reflects current registry state."""
    import asyncio
    app, registry = app_with_registry

    # Register a session directly into the registry
    async def _register():
        entry = SessionEntry(
            session_id="test-session-123",
            connected_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            has_voice_profile=False,
            owner=authkit.USER,
        )
        await registry.register(entry)

    asyncio.run(_register())

    client = TestClient(app)
    resp = client.get("/info", headers=authkit.auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["session_id"] == "test-session-123"
    assert data["sessions"][0]["has_voice_profile"] is False


def test_voice_upload_returns_404_for_invalid_session(app_with_registry):
    """Voice upload with non-existent session_id returns 404."""
    app, registry = app_with_registry
    client = TestClient(app)
    resp = client.post(
        "/voice/upload",
        params={**authkit.auth_params(), "session_id": "nonexistent"},
        headers=authkit.auth_headers(),
        files={"file": ("test.wav", b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100, "audio/wav")},
    )
    assert resp.status_code == 404
    assert "Session not found" in resp.json()["detail"]


def test_voice_upload_returns_401_for_unknown_bearer_token(app_with_registry):
    """Auth check on /voice/upload must return 401 for a token the store does not know."""
    app, registry = app_with_registry
    client = TestClient(app)
    resp = client.post(
        "/voice/upload",
        params={"session_id": "any"},
        headers=authkit.bearer_headers(authkit.UNKNOWN_TOKEN),
        files={"file": ("test.wav", b"data", "audio/wav")},
    )
    assert resp.status_code == 401


def test_voice_upload_returns_503_when_no_registry():
    """Voice upload returns 503 if session registry is not initialized."""
    app = create_app()
    # Do NOT attach session_registry to app.state (simulates registry not yet initialized)
    client = TestClient(app)
    resp = client.post(
        "/voice/upload",
        params={**authkit.auth_params(), "session_id": "any"},
        headers=authkit.auth_headers(),
        files={"file": ("test.wav", b"data", "audio/wav")},
    )
    assert resp.status_code == 503
    assert "Session registry not available" in resp.json()["detail"]
