"""Unit tests for STTPipeline buffer and VAD logic."""
import numpy as np
import pytest
from unittest.mock import MagicMock

import authkit  # noqa: F401  (test auth env; import before voicemanager)

from voicemanager.pipeline import STTPipeline, _pcm_to_float32, SAMPLE_RATE, VAD_CHUNK_BYTES, MAX_BUFFER_SAMPLES


@pytest.fixture
def mock_vad_model():
    """Mock Silero VAD model that always returns tensor(0.0) (no speech detected).

    VADIterator calls model(x, sampling_rate) and compares result to threshold float.
    Must return a tensor with a float .item() value.
    """
    import torch
    mock = MagicMock()
    mock.return_value = torch.tensor(0.0)
    return mock


@pytest.fixture
def mock_whisper_model():
    """Mock faster-whisper model."""
    return MagicMock()


@pytest.fixture
def mock_executor():
    """Mock ThreadPoolExecutor."""
    return MagicMock()


@pytest.fixture
def pipeline(mock_vad_model, mock_whisper_model, mock_executor):
    """STTPipeline with mock models."""
    return STTPipeline(
        vad_model=mock_vad_model,
        whisper_model=mock_whisper_model,
        executor=mock_executor,
        vad_threshold=0.5,
        silence_ms=500,
    )


def test_feed_returns_false_no_speech(pipeline):
    """feed() returns False when VAD detects no speech."""
    # Small chunk, no speech detected (VADIterator mock returns None)
    data = b"\x00\x00" * 256  # 512 bytes = 256 samples (< 512 sample VAD chunk)
    assert pipeline.feed(data) is False


def test_feed_accumulates_buffer(pipeline):
    """feed() accumulates data in internal buffer."""
    data1 = b"\x00\x01" * 100
    data2 = b"\x00\x02" * 200
    pipeline.feed(data1)
    pipeline.feed(data2)
    assert len(pipeline._buffer) == (100 + 200) * 2


def test_feed_triggers_on_max_buffer(pipeline):
    """feed() returns True when buffer exceeds 30s cap."""
    # MAX_BUFFER_SAMPLES = 16000 * 30 = 480000 samples = 960000 bytes
    big_data = b"\x00\x01" * MAX_BUFFER_SAMPLES  # exactly at limit
    result = pipeline.feed(big_data)
    assert result is True


def test_vad_leftover_carries_partial_chunks(pipeline):
    """Partial chunks < 512 samples are carried to next feed() call."""
    # 500 bytes = 250 samples (< 512 sample VAD chunk size of 1024 bytes)
    data = b"\x00\x01" * 250
    pipeline.feed(data)
    assert len(pipeline._vad_leftover) == 500  # carried over


def test_dispose_clears_state(pipeline):
    """dispose() clears buffer and leftover."""
    pipeline.feed(b"\x00\x01" * 100)
    pipeline.dispose()
    assert len(pipeline._buffer) == 0
    assert len(pipeline._vad_leftover) == 0


def test_pcm_to_float32_conversion():
    """_pcm_to_float32 normalizes int16 PCM to float32 [-1, 1]."""
    # Max positive int16 = 32767 -> ~1.0
    pcm = np.array([32767, -32768, 0], dtype=np.int16).tobytes()
    result = _pcm_to_float32(pcm)
    assert result.dtype == np.float32
    assert abs(result[0] - (32767 / 32768.0)) < 1e-5
    assert abs(result[1] - (-1.0)) < 1e-5
    assert result[2] == 0.0
