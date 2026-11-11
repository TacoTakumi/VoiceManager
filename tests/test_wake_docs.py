"""The server-side wake word is documented for client authors and server operators.

A client in any language must find in the client integration guide each wake frame
with every field of its model, a JSON example of it that the model accepts, the rules
of the dormant state, the rules for a phrase list, how a refusal looks and what the
two counts of the wake frame mean. The fields are read from the pydantic models, so a
field added to a frame fails this test until the guide names it. An operator must
find every wake setting in the guide's configuration table and in .env.example, with
the default the settings class has. The CHANGELOG must record the frames under the
release that adds them.

Model-free and server-free: it reads files and imports the protocol, config and wake
modules only.
"""
import json
import re
from pathlib import Path

import pytest

import authkit  # test auth env; must be imported before voicemanager  # noqa: F401

from voicemanager.config import Settings
from voicemanager.protocol import (
    ErrorMessage,
    MessageType,
    WakeMessage,
    WakeWordMessage,
    WakeWordReadyMessage,
)
from voicemanager.wake import MAX_PHRASE_CHARS, MAX_PHRASES

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE = (REPO_ROOT / "docs" / "client-integration.md").read_text(encoding="utf-8")
CHANGELOG = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
ENV_EXAMPLE = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

WAKE_HEADING = "## Server-side wake word"
WAKE_ANCHOR = "#server-side-wake-word"

CLIENT_FRAMES = (WakeWordMessage,)
SERVER_FRAMES = (WakeWordReadyMessage, WakeMessage)
FRAME_MODELS = CLIENT_FRAMES + SERVER_FRAMES

WAKE_SETTINGS = ("wake_engine", "wake_model_dir", "wake_phrases", "wake_threshold",
                 "wake_score")

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


def wake_section() -> str:
    return section(GUIDE, WAKE_HEADING)


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

def test_the_frame_models_are_the_wake_message_types():
    wake_types = {MessageType.wake_word, MessageType.wake_word_ready, MessageType.wake}
    assert {frame_type(m) for m in FRAME_MODELS} == {t.value for t in wake_types}


def test_the_guide_lists_the_section_in_its_contents():
    assert f"- [Server-side wake word]({WAKE_ANCHOR})" in GUIDE


@pytest.mark.parametrize("model", FRAME_MODELS, ids=frame_type)
def test_each_frame_is_in_the_schema_reference_with_every_field(model):
    body = frame_body(model)
    for name in model.model_fields:
        assert re.search(rf"^\| `{name}` \|", body, re.M), (
            f"{frame_type(model)}: the field table has no row for {name!r}")


@pytest.mark.parametrize("model", FRAME_MODELS, ids=frame_type)
def test_the_field_table_names_no_field_the_model_does_not_have(model):
    rows = re.findall(r"^\| `(\w+)` \|", frame_body(model), re.M)
    assert rows == list(model.model_fields)


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


def test_the_flow_example_shows_each_frame_as_its_model_has_it():
    """The frames in the section's flow diagram are real frames of the three types."""
    flow = re.search(r"```text\n(.*?)\n```", wake_section(), re.S).group(1)
    seen = set()
    for raw in re.findall(r"\{.*?\}(?=\s|$)", flow):
        frame = json.loads(raw)
        for model in FRAME_MODELS:
            if frame.get("type") == frame_type(model):
                model.model_validate(frame)
                seen.add(frame_type(model))
    assert seen == {frame_type(m) for m in FRAME_MODELS}


def test_the_section_links_each_frame_in_the_schema_reference():
    body = wake_section()
    for model in FRAME_MODELS:
        anchor = f"#{frame_type(model)}---{model.__name__.lower()}"
        assert f"[`{frame_type(model)}`]({anchor})" in body


# --- the dormant state ---------------------------------------------------------------

@pytest.mark.parametrize("rule", [
    "go to the spotter and to nothing else",
    "not sent to the STT or the turn engine",
    "not counted in `mic_sample`",
    "a `turn_engine` opt-in still succeeds after them",
    "sends one `wake` frame and listens",
    "The audio of the phrase itself is not transcribed",
    "`listen_stop` makes an awake connection dormant again",
    "`listen_start` wakes a dormant connection by hand",
    "The server sends no `wake` frame for it",
    "A second `wake_word` replaces the phrases",
    "exempt from `VM_SESSION_TIMEOUT`",
    "16 kHz mono 16-bit PCM",
    "no WAV header",
    "A client that never sends `wake_word` sees no change",
])
def test_the_section_gives_the_dormant_state_rules(rule):
    assert rule in squash(wake_section())


def test_the_section_says_a_client_with_its_own_spotter_does_not_need_it():
    body = squash(wake_section())
    assert "A client with its own on-device spotter does not need this" in body


# --- the phrases ---------------------------------------------------------------------

def test_the_phrase_limits_in_the_guide_are_the_limits_in_the_code():
    body = squash(section(wake_section(), "### The phrases"))
    assert f"1 to {MAX_PHRASES} phrases" in body
    assert f"each of {MAX_PHRASE_CHARS} characters or less" in body
    assert "letters, spaces and apostrophes" in body
    reference = squash(frame_body(WakeWordMessage))
    assert f"1 to {MAX_PHRASES} phrases of letters, spaces and apostrophes" in reference


@pytest.mark.parametrize("rule", [
    "the connection gets the server's default list, `VM_WAKE_PHRASES`",
    "It applies to that connection only",
    "lowercases each phrase",
    "leaves out a repeated phrase",
    "No model is trained for it",
    "A client cannot set them",
])
def test_the_section_gives_the_phrase_rules(rule):
    assert rule in squash(wake_section())


# --- a refusal -----------------------------------------------------------------------

def test_the_refusal_example_is_an_error_frame_the_client_can_match():
    body = section(wake_section(), "### A refused `wake_word`")
    examples = [json.loads(block) for block in JSON_FENCE.findall(body)]
    assert len(examples) == 1
    frame = ErrorMessage.model_validate(examples[0])
    assert frame.error_type == "invalid_request"
    assert frame.message.startswith("wake_word refused")
    text = squash(body)
    assert "The `message` starts with `wake_word refused`" in text
    assert "changes nothing" in text
    assert "`VM_WAKE_ENGINE=none`" in text
    assert "older than 0.9.0 drops the frame with no answer" in text


def test_the_refusal_prefix_in_the_guide_is_the_one_the_server_and_the_client_use():
    from voicemanager import client
    from voicemanager.transport import websocket

    assert client._WAKE_REFUSAL == "wake_word refused"
    source = Path(websocket.__file__).read_text(encoding="utf-8")
    assert 'message=f"wake_word refused: {reason}' in source


# --- the counts ----------------------------------------------------------------------

def test_the_section_says_what_the_two_counts_are_and_what_a_client_does_with_them():
    body = squash(section(wake_section(), "### The counts in the `wake` frame"))
    assert "Dormant audio is not in it" in body
    assert "up to and including the frame the phrase was heard in" in body
    assert "starts at 0 at each `wake_word`" in body
    assert "does not start again at a `listen_stop` on a connection that is dormant" in body
    assert "adds the samples it sent in this dormant period beyond `dormant_sample`" in body
    assert "can ignore the field" in body


# --- the settings --------------------------------------------------------------------

def _shown(value) -> str:
    """A default as the documents write it."""
    return "" if value == "" else str(value)


@pytest.mark.parametrize("name", WAKE_SETTINGS)
def test_env_example_lists_each_wake_setting_with_its_default(name):
    default = Settings.model_fields[name].default
    assert f"# VM_{name.upper()}={_shown(default)}\n" in ENV_EXAMPLE


def test_the_wake_settings_are_all_the_settings_named_wake():
    assert tuple(n for n in Settings.model_fields if n.startswith("wake_")) == WAKE_SETTINGS


@pytest.mark.parametrize("name", [n for n in WAKE_SETTINGS if n != "wake_model_dir"])
def test_the_configuration_table_has_each_wake_setting_with_its_default(name):
    table = section(GUIDE, "## Configuration")
    default = Settings.model_fields[name].default
    assert re.search(rf"^\| `VM_{name.upper()}` \| `{re.escape(str(default))}` \|", table, re.M)


def test_the_guide_and_the_env_example_say_how_to_install_and_enable_the_engine():
    for text in (squash(GUIDE), squash(ENV_EXAMPLE.replace("#", ""))):
        assert "requirements-wake.txt" in text
        assert "python setup_models.py wake" in text
        assert "VM_WAKE_MODEL_DIR" in text
    assert "`VM_WAKE_ENGINE=sherpa`" in squash(wake_section())


def test_the_python_client_helpers_are_in_the_method_reference():
    reference = section(GUIDE, "## Client method reference")
    for name in ("enable_wake_word", "wakes", "dormant", "wake_phrases"):
        assert re.search(rf"^\| `DuplexSession\.{name}` \|", reference, re.M), name
    assert "`WakeWordError`" in reference


# --- the changelog -------------------------------------------------------------------

def test_the_changelog_records_the_frames_under_0_9_0():
    entry = squash(section(CHANGELOG, "## [0.9.0]"))
    assert "### Added" in section(CHANGELOG, "## [0.9.0]")
    for name in ("`wake_word`", "`wake_word_ready`", "`wake`", "`VM_WAKE_ENGINE`",
                 "`VM_WAKE_MODEL_DIR`", "`VM_WAKE_PHRASES`", "`VM_WAKE_THRESHOLD`",
                 "`VM_WAKE_SCORE`", "`DuplexSession.enable_wake_word()`",
                 "`DuplexSession.wakes()`", "requirements-wake.txt",
                 "python setup_models.py wake"):
        assert name in entry, name
    assert "A client that does not send `wake_word` sees no change" in entry


def test_the_changelog_entry_is_above_the_last_release():
    assert CHANGELOG.index("## [0.9.0]") < CHANGELOG.index("## [0.8.4]")


def test_the_new_text_is_ascii():
    """Project documents are plain ASCII."""
    for name, text in (("the wake section", wake_section()),
                       ("the changelog entry", section(CHANGELOG, "## [0.9.0]"))):
        bad = sorted({ch for ch in text if ord(ch) > 127})
        assert not bad, f"{name} has non-ASCII characters: {bad}"
