"""Test configuration - sys.path for worktrees, the suite auth store, and the manual/automated split."""
import sys
import os

# When tests run from a git worktree, the editable install may point to the
# main repo. Prepend the worktree's src directory so worktree changes are visible.
_worktree_src = os.path.join(os.path.dirname(__file__), "..", "src")
_worktree_src = os.path.abspath(_worktree_src)
if _worktree_src not in sys.path:
    sys.path.insert(0, _worktree_src)

# Point VM_DB_PATH at a seeded temporary auth store (one user, one bearer token)
# before any test module imports voicemanager and caches its settings. Test files
# authenticate through authkit's helpers; see tests/authkit.py.
import authkit  # noqa: E402,F401


# Human-driven scripts, NOT part of the automated suite. They are named test_*.py
# for historical reasons, but they are smoke/demo tools you run by hand against a
# LIVE server (VoiceManager must already be up) and, for the mic one, real audio
# hardware. Most of them call asyncio.run(...) at module scope, so pytest merely
# COLLECTING them executes the script — which is why they must be ignored here
# rather than skipped with a marker (a marker cannot stop the import).
#
# Run them the way they are meant to be run, with a bearer token from
# 'voicemanager user token add' exported in the shell. Do not put it in .env:
# the server refuses to start on a key in .env that it does not know.
#     export VOICEMANAGER_TOKEN=vmp_...
#     python tests/test_tts.py
#     python tests/test_mic_stt.py --tts
#
# Automated tests are everything else in this directory: they are model-free and
# server-free (fakes/TestClient) and must stay that way. If you add a script that
# needs a live server, a GPU, a mic, or a human listening to the output, add it
# here — do not let it into the default `pytest tests/` run.
collect_ignore = [
    "test_tts.py",           # live server: synthesizes a WAV to disk
    "test_stt.py",           # live server: transcribes a fixture clip
    "test_stream_tts.py",    # live server: streaming TTS, writes WAVs
    "test_vibevoice_tts.py", # live server: VibeVoice voice sampling
    "test_mic_stt.py",       # live server + microphone: real-time dictation
]
