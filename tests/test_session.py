"""Unit tests for SessionRegistry and SessionEntry."""
import asyncio
import pytest
from datetime import datetime, timezone

from voicemanager.session import SessionEntry, SessionRegistry


def make_entry(session_id: str = "test-1") -> SessionEntry:
    now = datetime.now(timezone.utc)
    return SessionEntry(
        session_id=session_id,
        connected_at=now,
        last_activity=now,
    )


@pytest.mark.asyncio
async def test_register_and_get():
    """Test 1: register() adds a SessionEntry, get() retrieves it by session_id."""
    registry = SessionRegistry()
    entry = make_entry("sess-1")
    await registry.register(entry)
    result = await registry.get("sess-1")
    assert result is not None
    assert result.session_id == "sess-1"


@pytest.mark.asyncio
async def test_remove_deletes_session():
    """Test 2: remove() deletes a session, get() returns None after removal."""
    registry = SessionRegistry()
    entry = make_entry("sess-2")
    await registry.register(entry)
    await registry.remove("sess-2")
    result = await registry.get("sess-2")
    assert result is None


@pytest.mark.asyncio
async def test_remove_nonexistent_does_not_raise():
    """Test 3: remove() on non-existent session_id does not raise."""
    registry = SessionRegistry()
    # Should not raise any exception
    await registry.remove("nonexistent-id")


@pytest.mark.asyncio
async def test_update_activity_updates_timestamp():
    """Test 4: update_activity() updates last_activity timestamp on existing session."""
    registry = SessionRegistry()
    entry = make_entry("sess-4")
    original_activity = entry.last_activity
    await registry.register(entry)
    # Small sleep to ensure time has advanced
    await asyncio.sleep(0.01)
    await registry.update_activity("sess-4")
    updated = await registry.get("sess-4")
    assert updated is not None
    assert updated.last_activity >= original_activity


@pytest.mark.asyncio
async def test_update_activity_nonexistent_does_not_raise():
    """Test 5: update_activity() on non-existent session_id does not raise."""
    registry = SessionRegistry()
    # Should not raise any exception
    await registry.update_activity("nonexistent-id")


@pytest.mark.asyncio
async def test_set_voice_profile_sets_flag():
    """Test 6: set_voice_profile() sets has_voice_profile=True on session entry."""
    registry = SessionRegistry()
    entry = make_entry("sess-6")
    assert entry.has_voice_profile is False
    await registry.register(entry)
    await registry.set_voice_profile("sess-6", True)
    result = await registry.get("sess-6")
    assert result is not None
    assert result.has_voice_profile is True


@pytest.mark.asyncio
async def test_snapshot_returns_list_of_dicts():
    """Test 7: snapshot() returns list of dicts with required keys."""
    registry = SessionRegistry()
    entry = make_entry("sess-7")
    await registry.register(entry)
    snapshots = await registry.snapshot()
    assert len(snapshots) == 1
    snap = snapshots[0]
    assert "session_id" in snap
    assert "connected_at" in snap
    assert "last_activity" in snap
    assert "has_voice_profile" in snap
    assert snap["session_id"] == "sess-7"
    assert snap["has_voice_profile"] is False
    # connected_at and last_activity should be ISO format strings
    assert isinstance(snap["connected_at"], str)
    assert isinstance(snap["last_activity"], str)


@pytest.mark.asyncio
async def test_snapshot_empty_when_no_sessions():
    """Test 8: snapshot() returns empty list when no sessions registered."""
    registry = SessionRegistry()
    snapshots = await registry.snapshot()
    assert snapshots == []
