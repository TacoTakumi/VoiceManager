"""The listening boundary is documented for client authors.

A client in any language must find in the client integration guide each listen
frame with every field of its model, a JSON example of it that the model accepts,
the order in which to stop a reply and then stop listening, the rules while a
connection is stopped and the session timeout exemption. The fields are read from
the pydantic models, so a field added to a frame fails this test until the guide
names it. The CHANGELOG must record the frames under the release that adds them.

Model-free and server-free: it reads files and imports the protocol module only.
"""
import json
import re
from pathlib import Path

import pytest

from voicemanager.protocol import (
    ListenStartMessage,
    ListenStoppedMessage,
    ListenStopMessage,
    MessageType,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE = (REPO_ROOT / "docs" / "client-integration.md").read_text(encoding="utf-8")
CHANGELOG = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

LISTEN_HEADING = "## Listening boundary: stop and start listening on one connection"
LISTEN_ANCHOR = "#listening-boundary-stop-and-start-listening-on-one-connection"

CLIENT_FRAMES = (ListenStopMessage, ListenStartMessage)
SERVER_FRAMES = (ListenStoppedMessage,)
FRAME_MODELS = CLIENT_FRAMES + SERVER_FRAMES

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


def listen_section() -> str:
    return section(GUIDE, LISTEN_HEADING)


def frame_heading(model) -> str:
    return f"#### `{frame_type(model)}` - `{model.__name__}`"


def frame_body(model) -> str:
    direction = ("### Client to server frames" if model in CLIENT_FRAMES
                 else "### Server to client frames")
    reference = section(GUIDE, "## WebSocket message-schema reference")
    return section(section(reference, direction), frame_heading(model))


def squash(text: str) -> str:
    """The text with each run of white space made one space, so a phrase can wrap."""
    return " ".join(text.split())


# --- the frames ----------------------------------------------------------------------

def test_the_frame_models_are_the_listen_message_types():
    listen_types = {MessageType.listen_stop, MessageType.listen_start,
                    MessageType.listen_stopped}
    assert {frame_type(m) for m in FRAME_MODELS} == {t.value for t in listen_types}


def test_the_guide_lists_the_section_in_its_contents():
    assert f"- [Listening boundary: stop and start listening on one connection]" \
           f"({LISTEN_ANCHOR})" in GUIDE


@pytest.mark.parametrize("model", FRAME_MODELS, ids=frame_type)
def test_each_frame_is_in_the_schema_reference_with_every_field(model):
    body = frame_body(model)
    for name in model.model_fields:
        assert re.search(rf"^\| `{name}` \|", body, re.M), (
            f"{frame_type(model)}: the field table has no row for {name!r}")


@pytest.mark.parametrize("model", FRAME_MODELS, ids=frame_type)
def test_each_frame_has_a_json_example_the_model_accepts(model):
    examples = [json.loads(block) for block in JSON_FENCE.findall(frame_body(model))]
    ours = [e for e in examples if e.get("type") == frame_type(model)]
    assert ours, f"{frame_type(model)}: no JSON example of the frame"
    for example in ours:
        model.model_validate(example)


@pytest.mark.parametrize("model", FRAME_MODELS, ids=frame_type)
def test_every_json_example_of_a_frame_anywhere_in_the_guide_parses(model):
    for block in JSON_FENCE.findall(GUIDE):
        example = json.loads(block)
        if example.get("type") == frame_type(model):
            model.model_validate(example)


def test_the_section_links_the_frames_and_names_the_mic_sample_count():
    body = squash(listen_section())
    for model in FRAME_MODELS:
        assert f"`{frame_type(model)}`" in body
    assert "WAV header" in body
    assert "no answer" in body


# --- the contract ---------------------------------------------------------------------

def test_the_stop_order_is_playback_stop_then_tts_cancel_then_mic_off_then_listen_stop():
    body = section(listen_section(), "### Stopping in the middle of a reply")
    steps = [line for line in body.splitlines() if re.match(r"^\d\. ", line)]
    assert len(steps) == 4, f"expected four numbered steps, got {steps}"
    for step, needle in zip(steps, ("`playback` `stop`", "`tts_cancel`",
                                    "Stop the mic", "`listen_stop`")):
        assert needle in step, f"step {step!r} does not name {needle!r}"


@pytest.mark.parametrize("rule", [
    "dropped with no error",
    "not counted",
    "idempotent",
    "Neither frame ever sends an `error` frame",
    "extra fields are ignored",
    "A connection starts listening",
    "Older servers drop",
    "before the first audio frame",
    "need not wait for `listen_stopped`",
])
def test_the_section_gives_the_stopped_state_rules(rule):
    assert rule in squash(listen_section())


@pytest.mark.parametrize("kept", [
    "calibration", "VoxLock enrollment", "`turn_id` counter", "opt-in",
])
def test_the_section_says_what_a_stop_keeps(kept):
    body = squash(section(listen_section(), "### What the server does at `listen_stop`"))
    assert kept in body


@pytest.mark.parametrize("dropped", [
    "partial or final", "open utterance", "pending finals", "`turn_end`", "mic hold",
])
def test_the_section_says_what_a_stop_drops(dropped):
    body = squash(section(listen_section(), "### What the server does at `listen_stop`"))
    assert dropped in body


def test_the_section_says_tts_goes_on_while_stopped():
    body = squash(listen_section())
    assert "does not cancel TTS" in body
    assert "not tracked for barge" in body


def test_the_timeout_exemption_is_documented():
    body = squash(section(listen_section(), "### The session timeout"))
    assert "VM_SESSION_TIMEOUT" in body and "no `timeout_warning`" in body
    assert "from that frame" in body
    close_codes = squash(section(GUIDE, "## Close codes"))
    assert "stopped listening" in close_codes and "`listen_stop`" in close_codes


def test_the_audiocpp_tail_decode_and_the_late_barge_are_noted():
    body = squash(listen_section())
    assert "tail decode" in body
    assert "after `listen_stopped`" in body


# --- the CHANGELOG ----------------------------------------------------------------------

def _changelog_sections() -> list[tuple[str, str]]:
    parts = re.split(r"^## \[", CHANGELOG, flags=re.M)[1:]
    return [(p.partition("]")[0], p.partition("]")[2]) for p in parts]


def test_the_changelog_records_the_frames_under_0_8_0():
    sections = dict(_changelog_sections())
    assert "0.8.0" in sections, "no 0.8.0 section in CHANGELOG.md"
    body = sections["0.8.0"]
    assert "### Added" in body
    added = section("### Added" + body.split("### Added", 1)[1], "### Added")
    for model in FRAME_MODELS:
        assert f"`{frame_type(model)}`" in added
    assert "VM_SESSION_TIMEOUT" in added
