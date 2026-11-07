"""The turn engine is documented for client authors and for operators.

The client integration guide must give a non-Python client what it needs: each
turn-engine frame with every field of its model, a JSON example of it that the
model accepts, the error types the turn engine adds, and the audio_end reason
'interrupted'. The fields are read from the pydantic models, so a field added to a
frame fails this test until the guide names it. The README must tell an operator
how to set the sidecar up and which licences the model bundle carries, and the
CHANGELOG must record the feature.

Model-free and server-free: it reads files and imports the protocol module only.
"""
import json
import re
from pathlib import Path

import pytest

from voicemanager.protocol import (
    INVALID_REQUEST,
    TURN_ENGINE_ERROR_TYPES,
    BargeMessage,
    MessageType,
    PlaybackMessage,
    TurnEndMessage,
    TurnEngineMessage,
    TurnEngineReadyMessage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE = (REPO_ROOT / "docs" / "client-integration.md").read_text(encoding="utf-8")
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
CHANGELOG = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

FRAME_MODELS = (TurnEngineMessage, TurnEngineReadyMessage, PlaybackMessage,
                BargeMessage, TurnEndMessage)

JSON_FENCE = re.compile(r"```json\n(.*?)\n```", re.S)


def frame_type(model) -> str:
    return model.model_fields["type"].default


def section(text: str, heading: str) -> str:
    """The body under the first heading line equal to ``heading``, to the next
    heading of the same or a higher level. A '#' line inside a code fence (a shell
    comment) is not a heading."""
    level = len(heading) - len(heading.lstrip("#"))
    lines = text.splitlines()
    try:
        start = lines.index(heading)
    except ValueError:
        pytest.fail(f"no heading {heading!r}")
    body = []
    in_fence = False
    for line in lines[start + 1:]:
        if line.startswith("```"):
            in_fence = not in_fence
        elif line.startswith("#") and not in_fence:
            hashes = len(line) - len(line.lstrip("#"))
            if hashes <= level and line[hashes:hashes + 1] == " ":
                break
        body.append(line)
    return "\n".join(body)


def turn_engine_section() -> str:
    return section(GUIDE, "## Turn engine")


def frame_heading(model) -> str:
    return f"#### `{frame_type(model)}` - `{model.__name__}`"


def test_the_frame_models_are_the_new_message_types():
    new_types = {MessageType.turn_engine, MessageType.turn_engine_ready,
                 MessageType.playback, MessageType.barge, MessageType.turn_end}
    assert {frame_type(m) for m in FRAME_MODELS} == {t.value for t in new_types}


def test_the_guide_lists_the_turn_engine_section_in_its_contents():
    assert "- [Turn engine](#turn-engine)" in GUIDE


@pytest.mark.parametrize("model", FRAME_MODELS, ids=frame_type)
def test_each_frame_has_a_heading_with_every_field_in_its_table(model):
    body = section(turn_engine_section(), frame_heading(model))
    for name in model.model_fields:
        assert re.search(rf"^\| `{name}` \|", body, re.M), (
            f"{frame_type(model)}: the field table has no row for {name!r}")


@pytest.mark.parametrize("model", FRAME_MODELS, ids=frame_type)
def test_each_frame_has_a_json_example_the_model_accepts(model):
    body = section(turn_engine_section(), frame_heading(model))
    examples = [json.loads(block) for block in JSON_FENCE.findall(body)]
    ours = [e for e in examples if e.get("type") == frame_type(model)]
    assert ours, f"{frame_type(model)}: no JSON example of the frame"
    for example in ours:
        model.model_validate(example)


def test_playback_has_a_start_and_a_stop_example():
    body = section(turn_engine_section(), frame_heading(PlaybackMessage))
    events = {json.loads(b).get("event") for b in JSON_FENCE.findall(body)}
    assert {"start", "stop"} <= events


@pytest.mark.parametrize("error_type", TURN_ENGINE_ERROR_TYPES)
def test_the_guide_names_each_turn_engine_error(error_type):
    assert f"`{error_type}`" in turn_engine_section()
    assert f"`{error_type}`" in section(GUIDE, "## Close codes")


def test_the_guide_gives_the_invalid_request_cases_for_these_frames():
    body = turn_engine_section()
    assert f"`{INVALID_REQUEST}`" in body
    for case in ("before the first audio frame", "stop without a start",
                 "unknown `request_id`"):
        assert case in body, f"no invalid_request case {case!r}"


def test_the_guide_documents_interrupted_as_an_emitted_reason():
    assert '"interrupted"' in section(GUIDE, "#### `audio_end` - `AudioEndMessage`")
    assert '"interrupted"' in section(
        GUIDE, "### 3. Why an audio stream ended: `reason` on `audio_end`")
    assert "never emitted today" not in GUIDE
    assert "(reserved) `\"interrupted\"`" not in GUIDE


def test_the_guide_covers_the_mic_hold_and_losing_the_sidecar():
    body = turn_engine_section()
    for heading in ("### The mic hold", "### Losing the sidecar"):
        assert heading in body


@pytest.mark.parametrize("needle", [
    "OpenMDW-1.1", "NVIDIA", "venv-tyger", "requirements-tyger.txt",
    "setup_models.py tyger", "VM_TURN_ENGINE", "VM_TYGER_PY", "VM_TYGER_DEVICE",
    "Apache-2.0", "NOTICE", "audiocpp",
])
def test_the_readme_covers_the_tyger_setup(needle):
    assert needle in section(README, "### Tyger")


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def test_the_changelog_records_the_turn_engine_under_a_minor_release():
    sections = re.split(r"^## \[", CHANGELOG, flags=re.M)[1:]
    found = []
    for body in sections:
        version, _, rest = body.partition("]")
        added = section("### Added" + rest.split("### Added", 1)[1], "### Added") \
            if "### Added" in rest else ""
        if "VM_TURN_ENGINE" in added and "`turn_engine`" in added:
            found.append(version)
    assert found, "no Added entry for the turn engine in CHANGELOG.md"
    # New user-facing capability: a minor bump over the 0.4 releases.
    assert _version(found[0]) >= (0, 5, 0)
    assert "`interrupted`" in CHANGELOG


@pytest.mark.parametrize("name,text", [
    ("README.md", README), ("docs/client-integration.md", GUIDE),
    ("CHANGELOG.md", CHANGELOG),
])
def test_the_docs_are_ascii(name, text):
    bad = [n for n, line in enumerate(text.splitlines(), 1) if not line.isascii()]
    assert not bad, f"{name}: non-ASCII on lines {bad}"


# --- the experimental pre-filter ---------------------------------------------------------

ENV_EXAMPLE = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
CHANGELOG = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
PREFILTER_HEADING = "### Barge-in on speakers: the experimental pre-filter"


@pytest.mark.parametrize("name", ["VM_TYGER_PREFILTER", "VM_TYGER_PREFILTER_FLOOR_DB"])
def test_the_prefilter_settings_are_documented_as_experimental(name):
    env_block = ENV_EXAMPLE[ENV_EXAMPLE.index("# Experimental pre-filter"):]
    assert f"# {name}=" in env_block
    guide = section(turn_engine_section(), PREFILTER_HEADING)
    assert name in guide or name in README
    for text in (env_block, guide, README):
        assert "xperimental" in text


def test_the_guide_says_how_the_prefilter_enrolls_and_when_it_filters():
    guide = section(turn_engine_section(), PREFILTER_HEADING)
    assert "first 3 s" in guide and "while a reply plays" in guide
    assert "nothing from the client" in guide


def test_the_readme_names_the_install_and_the_setup_step():
    assert "onnxruntime-gpu" in README and "python setup_models.py tyger" in README
    assert "VM_TYGER_PREFILTER=voxlock" in README


def _release_section(version):
    """The CHANGELOG section of the release that shipped these settings."""
    start = CHANGELOG.index(f"## [{version}]")
    return CHANGELOG[start:CHANGELOG.index("## [", start + 4)]


def test_the_changelog_has_the_prefilter_under_the_next_version():
    newest = _release_section("0.8.0")
    assert "VM_TYGER_PREFILTER" in newest and "xperimental" in newest


@pytest.mark.parametrize("name", ["VM_TYGER_GATE_MARGIN_DB", "VM_TYGER_GATE_MAX_DB"])
def test_the_calibrated_gate_settings_are_documented(name):
    assert name in README and name in GUIDE
    assert f"# {name}=" in ENV_EXAMPLE
    newest = _release_section("0.8.0")
    assert name in newest

