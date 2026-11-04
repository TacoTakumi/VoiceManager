"""Tests for the server-to-sidecar wire format of the turn engine.

The server sends the sidecar one binary message per 320 ms hop: a 16-byte header
(magic, version, mode, the mic sample at the hop's end) and then 5120 user samples
and 5120 agent samples, all little-endian int16. The sidecar answers with event
lines (one JSON object per line) for an endpoint, a barge or an error. These tests
pin the layout, the round trip and every rejection.

The sidecar runs from its own virtual environment and loads the wire module by file
path, so the module must import with the standard library alone. One test proves
that in a fresh, isolated interpreter that cannot import pydantic, FastAPI or numpy.

Model-free and server-free.
"""
import array
import json
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from voicemanager.pipeline.tyger import wire
from voicemanager.pipeline.tyger.wire import (
    HEADER_BYTES,
    HOP_BYTES,
    HOP_SAMPLES,
    MAGIC,
    MODE_LISTENING,
    MODE_SPEAKING,
    SAMPLE_RATE,
    VERSION,
    BargeEvent,
    EndpointEvent,
    ErrorEvent,
    Hop,
    format_event,
    pack_hop,
    parse_event_line,
    parse_events,
    unpack_hop,
)

WIRE_PATH = Path(wire.__file__)


def _ramp(start: int) -> array.array:
    """5120 distinct int16 samples, wrapping inside the int16 range."""
    return array.array("h", (((start + i) % 65536) - 32768 for i in range(HOP_SAMPLES)))


def _le_bytes(samples: array.array) -> bytes:
    out = array.array("h", samples)
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


# --- layout -----------------------------------------------------------------

def test_hop_constants():
    assert SAMPLE_RATE == 16000
    assert HOP_SAMPLES == 5120                     # 320 ms at 16 kHz
    assert HOP_SAMPLES * 1000 // SAMPLE_RATE == 320
    assert HEADER_BYTES == 16
    assert HOP_BYTES == HEADER_BYTES + 2 * HOP_SAMPLES * 2
    assert MAGIC == b"TYGR"
    assert VERSION == 1


def test_packed_header_layout_is_little_endian():
    data = pack_hop(MODE_SPEAKING, 0x0102030405, _ramp(0), _ramp(7))
    assert len(data) == HOP_BYTES
    magic, version, mode, reserved, mic_sample = struct.unpack_from("<4sBBHq", data, 0)
    assert magic == b"TYGR"
    assert version == 1
    assert mode == 1                               # speaking
    assert reserved == 0
    assert mic_sample == 0x0102030405
    assert data[HEADER_BYTES:HEADER_BYTES + 2 * HOP_SAMPLES] == _le_bytes(_ramp(0))
    assert data[HEADER_BYTES + 2 * HOP_SAMPLES:] == _le_bytes(_ramp(7))


def test_listening_mode_byte_is_zero():
    data = pack_hop(MODE_LISTENING, 5120, bytes(2 * HOP_SAMPLES), bytes(2 * HOP_SAMPLES))
    assert data[5] == 0


# --- round trip ---------------------------------------------------------------

@pytest.mark.parametrize("mode", [MODE_LISTENING, MODE_SPEAKING])
def test_hop_round_trips(mode):
    user, agent = _ramp(100), _ramp(30000)
    hop = unpack_hop(pack_hop(mode, 2**40 + 5120, user, agent))
    assert isinstance(hop, Hop)
    assert hop.mode == mode
    assert hop.mic_sample == 2**40 + 5120
    assert hop.user == _le_bytes(user)
    assert hop.agent == _le_bytes(agent)
    assert hop.user_samples() == user
    assert hop.agent_samples() == agent


def test_unpack_accepts_bytearray_and_memoryview():
    data = pack_hop(MODE_LISTENING, 10240, _ramp(1), _ramp(2))
    assert unpack_hop(bytearray(data)) == unpack_hop(data)
    assert unpack_hop(memoryview(data)) == unpack_hop(data)


def test_pack_accepts_raw_bytes_int_lists_and_numpy_int16():
    user = _ramp(5)
    agent = _ramp(9)
    expected = pack_hop(MODE_SPEAKING, 15360, user, agent)
    assert pack_hop(MODE_SPEAKING, 15360, _le_bytes(user), _le_bytes(agent)) == expected
    assert pack_hop(MODE_SPEAKING, 15360, list(user), list(agent)) == expected
    np_user = np.array(user, dtype=np.int16)
    np_agent = np.array(agent, dtype=np.int16)
    assert pack_hop(MODE_SPEAKING, 15360, np_user, np_agent) == expected
    # An explicit big-endian array still goes on the wire little-endian.
    assert pack_hop(MODE_SPEAKING, 15360, np_user.astype(">i2"), np_agent) == expected


def test_extreme_sample_values_survive():
    user = array.array("h", [-32768, 32767] * (HOP_SAMPLES // 2))
    hop = unpack_hop(pack_hop(MODE_LISTENING, 0, user, user))
    assert hop.user_samples() == user


# --- pack rejections -------------------------------------------------------

@pytest.mark.parametrize("count", [0, HOP_SAMPLES - 1, HOP_SAMPLES + 1, 2 * HOP_SAMPLES])
def test_pack_rejects_wrong_sample_counts(count):
    good = _ramp(0)
    wrong = array.array("h", [0] * count)
    with pytest.raises(ValueError):
        pack_hop(MODE_LISTENING, 0, wrong, good)
    with pytest.raises(ValueError):
        pack_hop(MODE_LISTENING, 0, good, wrong)


def test_pack_rejects_odd_or_short_byte_strings():
    with pytest.raises(ValueError):
        pack_hop(MODE_LISTENING, 0, bytes(2 * HOP_SAMPLES - 1), bytes(2 * HOP_SAMPLES))
    with pytest.raises(ValueError):
        pack_hop(MODE_LISTENING, 0, bytes(2 * HOP_SAMPLES), bytes(HOP_SAMPLES))


def test_pack_rejects_other_sample_types():
    good = _ramp(0)
    with pytest.raises(ValueError):
        pack_hop(MODE_LISTENING, 0, np.zeros(HOP_SAMPLES, dtype=np.float32), good)
    with pytest.raises(ValueError):
        pack_hop(MODE_LISTENING, 0, np.zeros(HOP_SAMPLES, dtype=np.int32), good)
    with pytest.raises(ValueError):
        pack_hop(MODE_LISTENING, 0, [40000] + [0] * (HOP_SAMPLES - 1), good)


@pytest.mark.parametrize("mode", ["", "talking", "LISTENING", 0, None, ["speaking"]])
def test_pack_rejects_unknown_modes(mode):
    with pytest.raises(ValueError):
        pack_hop(mode, 0, _ramp(0), _ramp(0))


@pytest.mark.parametrize("mic_sample", [-1, 2**63, 1.5, "5120", True])
def test_pack_rejects_bad_mic_samples(mic_sample):
    with pytest.raises(ValueError):
        pack_hop(MODE_LISTENING, mic_sample, _ramp(0), _ramp(0))


# --- unpack rejections -----------------------------------------------------

def _good() -> bytes:
    return pack_hop(MODE_SPEAKING, 5120, _ramp(0), _ramp(1))


def _with_header(**fields) -> bytes:
    values = {"magic": MAGIC, "version": VERSION, "mode": 1, "reserved": 0,
              "mic_sample": 5120}
    values.update(fields)
    header = struct.pack("<4sBBHq", values["magic"], values["version"], values["mode"],
                         values["reserved"], values["mic_sample"])
    return header + _good()[HEADER_BYTES:]


@pytest.mark.parametrize("length", [0, HEADER_BYTES, HOP_BYTES - 1, HOP_BYTES + 1])
def test_unpack_rejects_wrong_lengths(length):
    data = (_good() + b"\x00")[:length]
    with pytest.raises(ValueError):
        unpack_hop(data)


@pytest.mark.parametrize("fields", [
    {"magic": b"TYGX"},
    {"magic": b"\x00\x00\x00\x00"},
    {"version": 0},
    {"version": 2},
    {"mode": 2},
    {"mode": 255},
    {"reserved": 1},
    {"mic_sample": -1},
])
def test_unpack_rejects_bad_headers(fields):
    with pytest.raises(ValueError):
        unpack_hop(_with_header(**fields))


def test_unpack_rejects_text():
    with pytest.raises(ValueError):
        unpack_hop("TYGR" + "\x00" * (HOP_BYTES - 4))


# --- event lines ---------------------------------------------------------------

@pytest.mark.parametrize("by", ["model", "silence"])
def test_endpoint_event_round_trips(by):
    event = EndpointEvent(by=by, mic_sample=163840)
    line = format_event(event)
    assert "\n" not in line
    assert json.loads(line) == {"event": "endpoint", "by": by, "mic_sample": 163840}
    assert parse_event_line(line) == event


def test_barge_event_round_trips():
    event = BargeEvent(onset_mic_sample=80000, mic_sample=84480)
    line = format_event(event)
    assert json.loads(line) == {
        "event": "barge", "onset_mic_sample": 80000, "mic_sample": 84480}
    assert parse_event_line(line) == event


def test_error_event_round_trips():
    event = ErrorEvent(message="scorer failed: out of memory")
    line = format_event(event)
    assert json.loads(line) == {"event": "error", "message": "scorer failed: out of memory"}
    assert parse_event_line(line) == event


def test_error_event_with_a_newline_stays_on_one_line():
    line = format_event(ErrorEvent(message="first\nsecond"))
    assert "\n" not in line
    assert parse_event_line(line).message == "first\nsecond"


def test_parse_event_line_ignores_surrounding_whitespace_and_unknown_keys():
    line = ' {"event": "endpoint", "by": "model", "mic_sample": 5120, "score": 0.9}\n'
    assert parse_event_line(line) == EndpointEvent(by="model", mic_sample=5120)


def test_parse_events_reads_every_line_in_order_and_skips_blank_lines():
    text = "\n".join([
        format_event(EndpointEvent(by="silence", mic_sample=5120)),
        "",
        format_event(BargeEvent(onset_mic_sample=9000, mic_sample=10240)),
        format_event(ErrorEvent(message="x")),
        "",
    ])
    assert parse_events(text) == [
        EndpointEvent(by="silence", mic_sample=5120),
        BargeEvent(onset_mic_sample=9000, mic_sample=10240),
        ErrorEvent(message="x"),
    ]


@pytest.mark.parametrize("line", [
    "",
    "not json",
    "[]",
    '"endpoint"',
    '{"by": "model", "mic_sample": 1}',                                  # no event
    '{"event": "retract", "mic_sample": 1}',                             # unknown event
    '{"event": "endpoint", "mic_sample": 1}',                            # no by
    '{"event": "endpoint", "by": "vad", "mic_sample": 1}',               # unknown by
    '{"event": "endpoint", "by": "model"}',                              # no mic_sample
    '{"event": "endpoint", "by": "model", "mic_sample": -1}',
    '{"event": "endpoint", "by": "model", "mic_sample": 1.5}',
    '{"event": "endpoint", "by": "model", "mic_sample": "1"}',
    '{"event": "endpoint", "by": "model", "mic_sample": true}',
    '{"event": "barge", "mic_sample": 1}',                               # no onset
    '{"event": "barge", "onset_mic_sample": 1}',                         # no mic_sample
    '{"event": "barge", "onset_mic_sample": 10, "mic_sample": 5}',       # onset after
    '{"event": "barge", "onset_mic_sample": -1, "mic_sample": 5}',
    '{"event": "error"}',                                                # no message
    '{"event": "error", "message": 5}',
    '{"event": "endpoint", "by": "model", "mic_sample": 1}\n{"event": "error"}',
])
def test_parse_event_line_rejects_bad_lines(line):
    with pytest.raises(ValueError):
        parse_event_line(line)


def test_parse_events_rejects_a_bad_line_among_good_ones():
    text = format_event(EndpointEvent(by="model", mic_sample=1)) + "\nnot json\n"
    with pytest.raises(ValueError):
        parse_events(text)


def test_format_event_rejects_invalid_events():
    with pytest.raises(ValueError):
        format_event(EndpointEvent(by="vad", mic_sample=1))
    with pytest.raises(ValueError):
        format_event(BargeEvent(onset_mic_sample=10, mic_sample=5))
    with pytest.raises(ValueError):
        format_event("endpoint")


# --- stdlib-only import ----------------------------------------------------------

def test_wire_module_imports_with_the_standard_library_alone(tmp_path):
    # A fresh isolated interpreter loads wire.py by file path, as the sidecar does,
    # with pydantic, FastAPI, numpy, torch and the voicemanager package blocked.
    script = tmp_path / "load_wire.py"
    script.write_text(textwrap.dedent("""
        import importlib.abc
        import importlib.util
        import sys

        BLOCKED = {"pydantic", "fastapi", "numpy", "torch", "starlette", "voicemanager"}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError("blocked: " + name)
                return None

        sys.meta_path.insert(0, Blocker())
        spec = importlib.util.spec_from_file_location("tyger_wire", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        zeros = bytes(2 * module.HOP_SAMPLES)
        hop = module.unpack_hop(module.pack_hop(module.MODE_SPEAKING, 5120, zeros, zeros))
        assert hop.mode == module.MODE_SPEAKING and hop.mic_sample == 5120
        line = module.format_event(module.BargeEvent(onset_mic_sample=1, mic_sample=2))
        assert module.parse_event_line(line).mic_sample == 2
        print("ok")
    """))
    result = subprocess.run(
        [sys.executable, "-I", str(script), str(WIRE_PATH)],
        capture_output=True, text=True, cwd=tmp_path, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_tyger_package_init_imports_nothing_heavy():
    # The package __init__ must stay import-free so that importing the wire module
    # through the package pulls in no scorer, torch or transformers code of its own.
    source = (WIRE_PATH.parent / "__init__.py").read_text()
    imports = [line for line in source.splitlines()
               if line.strip().startswith(("import ", "from "))]
    assert imports == []
