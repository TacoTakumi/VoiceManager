"""Wrapper for the model-free sidecar drive-loop C++ tests (STT and TTS).

Runs the ``audiocpp_sidecar_tests`` binary - the REAL drive_streaming_session
AND drive_tts_session against fake IStreamingVoiceTaskSessions over socketpairs
(no model, no GPU, no engine libs) - and skips cleanly when no sidecar build
tree is present, so the automated suite stays green on a checkout without the
C++ toolchain.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_test_binary() -> Path | None:
    candidates = sorted(REPO_ROOT.glob("sidecar/build*/bin/audiocpp_sidecar_tests"))
    return candidates[-1] if candidates else None


def test_drive_loop_rearm() -> None:
    binary = _find_test_binary()
    if binary is None:
        pytest.skip("sidecar build tree absent; C++ drive-loop tests not built")
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"audiocpp_sidecar_tests failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
