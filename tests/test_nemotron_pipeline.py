"""Unit tests for the Nemotron cache-aware STT settings and pipeline logic."""

import authkit  # test auth env; import before voicemanager

from voicemanager.config import Settings


def test_nemotron_settings_have_documented_defaults():
    s = Settings()
    assert s.nemotron_model == "nvidia/nemotron-speech-streaming-en-0.6b"
    # Cache-aware latency mode (att_context_size = [left, right], in 80ms frames).
    assert isinstance(s.nemotron_att_context_left, int)
    assert isinstance(s.nemotron_att_context_right, int)
    # EoU endpointing thresholds.
    assert s.nemotron_eou_stop_history_ms > 0
    assert s.nemotron_eou_residue_tokens >= 0


def test_nemotron_settings_override_from_env(monkeypatch):
    monkeypatch.setenv("VM_NEMOTRON_ATT_CONTEXT_RIGHT", "1")
    monkeypatch.setenv("VM_NEMOTRON_EOU_STOP_HISTORY_MS", "800")
    s = Settings()
    assert s.nemotron_att_context_right == 1
    assert s.nemotron_eou_stop_history_ms == 800


# --- NemotronSTTPipeline wrapper logic (NeMo seam overridden, no model load) -------

import asyncio

from voicemanager.pipeline.nemotron_stt import NemotronSTTPipeline

# The wrapper buffers incoming PCM to the pipeline's chunk size and drives exactly one
# cache-aware transcribe step per full chunk (NeMo's bufferer takes the last chunk_size
# of its rolling buffer per step, so frames MUST be chunk-aligned — see nemotron_stt.py).
# Use a tiny 256-sample chunk so one feed() of _ONE_CHUNK delivers exactly one step.
_CHUNK_SAMPLES = 256
_ONE_CHUNK = b"\x00\x00" * _CHUNK_SAMPLES  # 256 int16 samples of silence == one chunk


class _FakeShared:
    """Stand-in for CacheAwareRNNTPipeline: init_state/delete_state + chunk size.

    Exposes chunk_size_in_secs (the wrapper reads it to size its buffer) and records
    delete_state() calls so dispose()/finalize() teardown can be asserted.
    """
    chunk_size_in_secs = _CHUNK_SAMPLES / 16000  # 0.016s -> chunk_samples == 256

    def __init__(self):
        self.deleted = []

    def init_state(self, stream_id, options):
        return object()

    def delete_state(self, stream_id):
        self.deleted.append(stream_id)


class _ScriptedNemotron(NemotronSTTPipeline):
    """Drives the wrapper with scripted (partial, final_or_None) outputs per chunk.

    Each scripted step is (partial_text, final_text_or_None); a non-None final means
    the endpointer fired an EoU on that step. _make_options,
    _transcribe and _finalize_sync are overridden so unit tests never import NeMo or
    load a model. `final_tail` is the residue the is_last flush returns from finalize().
    """
    def __init__(self, script, final_tail="", **kw):
        self._script = list(script)
        self._final_tail = final_tail
        super().__init__(**kw)

    def _make_options(self):
        return None  # fake init_state ignores it; avoids importing NeMo in unit tests

    def _transcribe(self, audio_f32):
        return self._script.pop(0) if self._script else ("", None)

    def _finalize_sync(self):
        # Mirror the real guard (is_last is sent once) without touching NeMo.
        if self._closed:
            return ""
        self._closed = True
        return self._final_tail


def test_feed_emits_interim_then_take_final_on_eou():
    # step 1: partial "this", no EoU. step 2: partial "this facility", EoU -> final.
    fake = _FakeShared()
    p = _ScriptedNemotron([("this", None), ("this facility", "this facility")],
                          shared_pipeline=fake, stream_id=1, att_context_size=[70, 13])

    assert asyncio.run(p.feed(_ONE_CHUNK)) is True   # interim available
    assert p.get_current_text() == ("this", "en", 0)
    assert p.take_final() is None  # no EoU yet

    assert asyncio.run(p.feed(_ONE_CHUNK)) is False  # EoU step: a final, not an interim
    final = p.take_final()
    assert final == ("this facility", "en", 0)   # final carries the utterance's segment_index
    assert p.take_final() is None                 # drained


def test_segment_index_increments_per_final():
    p = _ScriptedNemotron([("a", "a"), ("b", "b")],
                          shared_pipeline=_FakeShared(), stream_id=2, att_context_size=[70, 13])
    asyncio.run(p.feed(_ONE_CHUNK))
    assert p.take_final() == ("a", "en", 0)
    asyncio.run(p.feed(_ONE_CHUNK))
    assert p.take_final() == ("b", "en", 1)       # monotonic increment per final


def test_feed_buffers_subchunk_audio_until_a_full_chunk():
    # A frame shorter than one chunk must NOT drive a transcribe step yet (chunk-align).
    p = _ScriptedNemotron([("partial", None)],
                          shared_pipeline=_FakeShared(), stream_id=5, att_context_size=[70, 13])
    half = b"\x00\x00" * (_CHUNK_SAMPLES // 2)
    assert asyncio.run(p.feed(half)) is False     # buffered, no step -> no interim
    assert p.get_current_text() == ("", "en", 0)
    assert asyncio.run(p.feed(half)) is True       # second half completes a chunk -> one step
    assert p.get_current_text() == ("partial", "en", 0)


def test_dispose_deletes_stream_state():
    fake = _FakeShared()
    p = _ScriptedNemotron([], shared_pipeline=fake, stream_id=7, att_context_size=[70, 13])
    p.dispose()
    assert fake.deleted == [7]


def test_branch_emits_one_final_per_utterance():
    # Mirrors the websocket nemotron branch's emit decision: interims from
    # get_current_text() when feed() reports an update, one final per take_final().
    # interim "this", interim+EoU "this facility"; then interim "not", interim+EoU "not authorized".
    p = _ScriptedNemotron([
        ("this", None), ("this facility", "this facility"),
        ("not", None), ("not authorized", "not authorized"),
    ], shared_pipeline=_FakeShared(), stream_id=3, att_context_size=[70, 13])
    interims, finals = [], []
    for _ in range(4):
        if asyncio.run(p.feed(_ONE_CHUNK)):
            interims.append(p.get_current_text())
        while (f := p.take_final()) is not None:
            finals.append(f)
            p.reset()
    assert finals == [("this facility", "en", 0), ("not authorized", "en", 1)]
    assert ("this", "en", 0) in interims


# --- Engine selection / reporting (TestClient, lifespan skipped -> no model load) ----

from fastapi.testclient import TestClient


def test_info_reports_nemotron_engine(monkeypatch):
    monkeypatch.setenv("VM_STT_ENGINE", "nemotron")
    # Reset the cached settings singleton so the env override is picked up, and reset
    # again afterwards so the nemotron engine does not leak into other tests' settings.
    import voicemanager.config as cfg
    cfg._settings = None
    try:
        from voicemanager.app import create_app
        # TestClient context-manager is NOT used → lifespan skipped → no model load;
        # app.state attrs absent, so /info falls back to settings.stt_engine.
        client = TestClient(create_app())
        info = client.get("/info", headers=authkit.auth_headers()).json()
        assert info["stt_engine"] == "nemotron"
    finally:
        cfg._settings = None


def test_nemotron_cfg_disables_per_stream_biasing():
    """Regression: per-stream biasing drags in the ngram_multi_advance Triton kernel,
    which has no compiled image on Blackwell/sm_120 (RTX 5060 Ti) -> transcribe crashes
    with "Triton Error [CUDA]: no kernel image is available". We configure no biasing,
    so it MUST stay off. (Found in live verification on cuda:2.)"""
    import pytest
    pytest.importorskip("omegaconf")  # from NeMo (requirements-nemotron.txt)
    from voicemanager.models import _build_nemotron_cfg
    s = Settings()
    cfg = _build_nemotron_cfg(s, "cuda:2")
    assert cfg.asr.decoding.greedy.enable_per_stream_biasing is False
