"""Wire format between the server and the Tyger sidecar.

The sidecar runs from its own virtual environment and loads this file by path, so it
imports only the standard library: no pydantic, no FastAPI, no numpy. Both processes
use it, so the two ends cannot drift apart.

Each opted-in session has one WebSocket to the sidecar. A session that connects with
the query parameter ``progress=1`` (``PROGRESS_PARAM``) asks for progress lines (see
below).

Server to sidecar: one binary message per hop of 320 ms (5120 samples at 16 kHz).
All fields are little-endian::

    offset  size   field
    0       4      magic, the ASCII bytes "TYGR"
    4       1      version, uint8, currently 1
    5       1      mode, uint8: 0 listening (no reply plays), 1 speaking
    6       2      reserved, uint16, must be 0
    8       8      mic_sample, int64 >= 0: the index on the client's mic stream of
                   the sample just after the hop's last user sample (the hop's end)
    16      10240  user channel, 5120 int16 samples of mic audio
    10256   10240  agent channel, 5120 int16 samples of the played reply, silence
                   when nothing plays

A hop is exactly 20496 bytes. Anything else is rejected with ValueError.

Server to sidecar, control: one text message holding one JSON object with a
``control`` key. The one control defined is the listening boundary::

    {"control": "boundary"}

The server sends it when the client stops listening on the session (it stopped
sending mic audio). It tells the sidecar to drop the session's turn state built from
the hops before it; the session's WebSocket stays open. ``is_boundary_control``
recognises it: surrounding whitespace and unknown keys are allowed, as for event
lines. Any other text message (not JSON, not an object, more than one line, no
``control`` key, an unknown or ill-typed control) is not a control message.

Sidecar to server: text messages holding event lines, one JSON object per line
(a message may hold more than one line; blank lines are skipped)::

    {"event": "endpoint", "by": "model" | "silence", "mic_sample": int}
    {"event": "barge", "onset_mic_sample": int, "mic_sample": int}
    {"event": "error", "message": str}
    {"event": "scored", "mic_sample": int}
    {"event": "barge_state", "state": "learning" | "ready"}

``mic_sample`` is the hop end at which the sidecar decided; ``onset_mic_sample`` is
where the user's speech began, never after ``mic_sample``. Unknown keys are ignored
so that either end can add one; a missing or ill-typed known key is rejected with
ValueError.

A ``scored`` line is a progress line, sent only to a session that asked for them:
the sidecar is done with every hop up to the one that ends at ``mic_sample``, and
every event those hops give was sent before it. One goes out with ``mic_sample`` 0
as soon as the session opens (it says the sidecar sends them), and one after each
hop, last in that hop's message, whether the hop scored or failed with an error
line. A hop that is not a valid hop message gets none. So once the server has read
a scored line at X, it has read every barge and endpoint up to X.

A ``barge_state`` line comes only from a sidecar that runs the pre-filter. The
pre-filter gives no barge until it has learned the user's voice: ``learning`` goes
out when such a session opens, before its first scored line, and ``ready`` once, in
the message of the hop at which the pre-filter enrolled, before that hop's scored
line. A session with no pre-filter gets neither.
"""
import array
import json
import struct
import sys
from dataclasses import dataclass

SAMPLE_RATE = 16000
HOP_SAMPLES = 5120                      # 320 ms at 16 kHz
HOP_MS = HOP_SAMPLES * 1000 // SAMPLE_RATE
CHANNEL_BYTES = HOP_SAMPLES * 2         # int16

MAGIC = b"TYGR"
VERSION = 1
_HEADER = struct.Struct("<4sBBHq")
HEADER_BYTES = _HEADER.size             # 16
HOP_BYTES = HEADER_BYTES + 2 * CHANNEL_BYTES

MODE_LISTENING = "listening"
MODE_SPEAKING = "speaking"
_MODE_CODES = {MODE_LISTENING: 0, MODE_SPEAKING: 1}
_MODE_NAMES = {code: name for name, code in _MODE_CODES.items()}

ENDPOINT_BY: tuple[str, ...] = ("model", "silence")

BARGE_LEARNING = "learning"
BARGE_READY = "ready"
BARGE_STATES: tuple[str, ...] = (BARGE_LEARNING, BARGE_READY)

#: The session URL's query parameter that asks for progress lines, set to "1".
PROGRESS_PARAM = "progress"

#: The key that marks a server-to-sidecar text message as a control message.
CONTROL_KEY = "control"
#: The control value for a listening boundary on the session.
CONTROL_BOUNDARY = "boundary"
#: The boundary control message exactly as the server sends it.
BOUNDARY_CONTROL = json.dumps({CONTROL_KEY: CONTROL_BOUNDARY}, separators=(",", ":"))

_INT64_MAX = 2**63 - 1
_BIG_ENDIAN_HOST = sys.byteorder != "little"


@dataclass(frozen=True)
class Hop:
    """One unpacked hop. ``user`` and ``agent`` are little-endian int16 PCM bytes."""
    mode: str
    mic_sample: int
    user: bytes
    agent: bytes

    def user_samples(self) -> array.array:
        """The user channel as an int16 array in host order."""
        return _samples(self.user)

    def agent_samples(self) -> array.array:
        """The agent channel as an int16 array in host order."""
        return _samples(self.agent)


@dataclass(frozen=True)
class EndpointEvent:
    """The user's turn ended at ``mic_sample``, by the model or its silence fallback."""
    by: str
    mic_sample: int


@dataclass(frozen=True)
class BargeEvent:
    """The user cut in over the played reply: speech began at ``onset_mic_sample``."""
    onset_mic_sample: int
    mic_sample: int


@dataclass(frozen=True)
class ErrorEvent:
    """The sidecar failed on this session."""
    message: str


@dataclass(frozen=True)
class ScoredEvent:
    """Progress: the sidecar is done with every hop up to the one ending at mic_sample."""
    mic_sample: int


@dataclass(frozen=True)
class BargeStateEvent:
    """Whether the session's pre-filter can give a barge yet: ``learning`` until it
    has the user's voice, ``ready`` from then on."""
    state: str


Event = EndpointEvent | BargeEvent | ErrorEvent | ScoredEvent | BargeStateEvent


# --- hops ---------------------------------------------------------------------

def pack_hop(mode: str, mic_sample: int, user, agent) -> bytes:
    """Pack one hop into its binary message.

    ``user`` and ``agent`` each hold exactly 5120 int16 samples, given as raw
    little-endian PCM bytes (bytes, bytearray or a byte memoryview), as any int16
    buffer (array('h'), a numpy int16 array of either byte order), or as a sequence
    of ints in the int16 range.
    """
    if not isinstance(mode, str) or mode not in _MODE_CODES:
        raise ValueError(f"mode must be {MODE_LISTENING!r} or {MODE_SPEAKING!r}, got {mode!r}")
    _check_sample_index("mic_sample", mic_sample)
    header = _HEADER.pack(MAGIC, VERSION, _MODE_CODES[mode], 0, mic_sample)
    return header + _pcm16le(user, "user") + _pcm16le(agent, "agent")


def unpack_hop(data) -> Hop:
    """Unpack one binary hop message, checking its length and header."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"a hop is a binary message, got {type(data).__name__}")
    data = bytes(data)
    if len(data) != HOP_BYTES:
        raise ValueError(f"a hop is {HOP_BYTES} bytes, got {len(data)}")
    magic, version, mode_code, reserved, mic_sample = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad hop magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported hop version {version}")
    if mode_code not in _MODE_NAMES:
        raise ValueError(f"bad hop mode {mode_code}")
    if reserved != 0:
        raise ValueError(f"hop reserved field must be 0, got {reserved}")
    if mic_sample < 0:
        raise ValueError(f"hop mic_sample must be >= 0, got {mic_sample}")
    user = data[HEADER_BYTES:HEADER_BYTES + CHANNEL_BYTES]
    agent = data[HEADER_BYTES + CHANNEL_BYTES:]
    return Hop(mode=_MODE_NAMES[mode_code], mic_sample=mic_sample, user=user, agent=agent)


def _pcm16le(samples, name: str) -> bytes:
    """One channel as exactly CHANNEL_BYTES of little-endian int16 PCM."""
    try:
        view = memoryview(samples)
    except TypeError:
        view = None
    if view is None:
        # A sequence of ints.
        try:
            values = array.array("h", samples)
        except (TypeError, OverflowError) as exc:
            raise ValueError(f"{name} samples must be int16 values: {exc}") from None
        if len(values) != HOP_SAMPLES:
            raise ValueError(f"{name} needs {HOP_SAMPLES} samples, got {len(values)}")
        if _BIG_ENDIAN_HOST:
            values.byteswap()
        return values.tobytes()

    fmt = view.format
    if fmt in ("B", "b", "c"):
        # Raw bytes: taken as little-endian PCM already.
        data = view.tobytes()
        if len(data) != CHANNEL_BYTES:
            raise ValueError(
                f"{name} needs {CHANNEL_BYTES} bytes ({HOP_SAMPLES} int16 samples), "
                f"got {len(data)}")
        return data
    order, code = (fmt[0], fmt[1:]) if fmt[:1] in ("<", ">", "=", "@", "!") else ("@", fmt)
    if code != "h" or view.itemsize != 2:
        raise ValueError(f"{name} samples must be int16, got buffer format {fmt!r}")
    if view.nbytes != CHANNEL_BYTES:
        raise ValueError(f"{name} needs {HOP_SAMPLES} samples, got {view.nbytes // 2}")
    data = view.tobytes()
    big = order in (">", "!") or (order in ("=", "@") and _BIG_ENDIAN_HOST)
    if big:
        swapped = array.array("h", data)
        swapped.byteswap()
        data = swapped.tobytes()
    return data


def _samples(pcm: bytes) -> array.array:
    values = array.array("h", pcm)
    if _BIG_ENDIAN_HOST:
        values.byteswap()
    return values


def _check_sample_index(name: str, value) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    if not 0 <= value <= _INT64_MAX:
        raise ValueError(f"{name} must be in 0..2**63-1, got {value}")


# --- control messages -------------------------------------------------------------

def format_boundary_control() -> str:
    """The listening-boundary control as one JSON text message, without a newline."""
    return BOUNDARY_CONTROL


def is_boundary_control(message) -> bool:
    """Whether one text message is the listening-boundary control.

    Never raises: a binary message, a malformed text message or an unknown control
    is simply not the boundary control.
    """
    if not isinstance(message, str):
        return False
    text = message.strip()
    if not text or "\n" in text:
        return False
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict):
        return False
    value = obj.get(CONTROL_KEY)
    return type(value) is str and value == CONTROL_BOUNDARY


# --- event lines ----------------------------------------------------------------

def format_event(event: Event) -> str:
    """One event as a single JSON line, without the trailing newline."""
    _check_event(event)
    if isinstance(event, EndpointEvent):
        obj = {"event": "endpoint", "by": event.by, "mic_sample": event.mic_sample}
    elif isinstance(event, BargeEvent):
        obj = {"event": "barge", "onset_mic_sample": event.onset_mic_sample,
               "mic_sample": event.mic_sample}
    elif isinstance(event, ScoredEvent):
        obj = {"event": "scored", "mic_sample": event.mic_sample}
    elif isinstance(event, BargeStateEvent):
        obj = {"event": "barge_state", "state": event.state}
    else:
        obj = {"event": "error", "message": event.message}
    # json.dumps escapes a newline inside a string, so the line stays one line.
    return json.dumps(obj, separators=(",", ":"))


def parse_event_line(line: str) -> Event:
    """Parse one event line (surrounding whitespace allowed)."""
    if not isinstance(line, str):
        raise ValueError(f"an event line is text, got {type(line).__name__}")
    text = line.strip()
    if not text:
        raise ValueError("empty event line")
    if "\n" in text:
        raise ValueError("an event line holds one event; use parse_events for several")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"event line is not JSON: {exc}") from None
    if not isinstance(obj, dict):
        raise ValueError("an event line is a JSON object")
    kind = obj.get("event")
    if kind == "endpoint":
        event = EndpointEvent(by=_get(obj, "by", str), mic_sample=_get(obj, "mic_sample", int))
    elif kind == "barge":
        event = BargeEvent(onset_mic_sample=_get(obj, "onset_mic_sample", int),
                           mic_sample=_get(obj, "mic_sample", int))
    elif kind == "error":
        event = ErrorEvent(message=_get(obj, "message", str))
    elif kind == "scored":
        event = ScoredEvent(mic_sample=_get(obj, "mic_sample", int))
    elif kind == "barge_state":
        event = BargeStateEvent(state=_get(obj, "state", str))
    else:
        raise ValueError(f"unknown event {kind!r}")
    _check_event(event)
    return event


def parse_events(text: str) -> list[Event]:
    """Parse every event line in one message, in order, skipping blank lines."""
    if not isinstance(text, str):
        raise ValueError(f"event lines are text, got {type(text).__name__}")
    return [parse_event_line(line) for line in text.splitlines() if line.strip()]


def _get(obj: dict, key: str, kind: type):
    if key not in obj:
        raise ValueError(f"event {obj.get('event')!r} needs {key!r}")
    value = obj[key]
    # bool is an int subclass; a JSON true is not a sample index.
    if type(value) is not kind:
        raise ValueError(f"event field {key!r} must be {kind.__name__}, got {value!r}")
    return value


def _check_event(event) -> None:
    if isinstance(event, EndpointEvent):
        if event.by not in ENDPOINT_BY:
            raise ValueError(f"endpoint by must be one of {ENDPOINT_BY}, got {event.by!r}")
        _check_sample_index("mic_sample", event.mic_sample)
    elif isinstance(event, BargeEvent):
        _check_sample_index("onset_mic_sample", event.onset_mic_sample)
        _check_sample_index("mic_sample", event.mic_sample)
        if event.onset_mic_sample > event.mic_sample:
            raise ValueError(
                f"barge onset_mic_sample {event.onset_mic_sample} is after "
                f"mic_sample {event.mic_sample}")
    elif isinstance(event, ErrorEvent):
        if not isinstance(event.message, str):
            raise ValueError("error message must be a string")
    elif isinstance(event, ScoredEvent):
        _check_sample_index("mic_sample", event.mic_sample)
    elif isinstance(event, BargeStateEvent):
        if event.state not in BARGE_STATES:
            raise ValueError(
                f"barge_state state must be one of {BARGE_STATES}, got {event.state!r}")
    else:
        raise ValueError(f"not a turn-engine event: {event!r}")
