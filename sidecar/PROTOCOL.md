# audio.cpp sidecar wire protocol (v1)

Status: v1, implemented - the handshake, the AUDIO/EVENT/FLUSH/FINAL/RESET streaming
drive loop, and process-fatal-fault containment + the ERROR `sidecar_fault` signal.
The additive TTS frame set (TEXT/TTS_AUDIO/TTS_DONE, still protocol 1) and its drive
path are added by audiocpp-tts-backend. voxcpm2-tts-streaming (a second TTS
model_task) is added by audiocpp-voxcpm2-tts, with per-model ref_text and its
`retry_badcase=false` request-option pin; the audio.cpp v0.7.3 pin bump added
`voxcpm2.chunk_strategy=stateless` beside it. breeze-tts-streaming (a third TTS
model_task, 24 kHz output) is added by audiocpp-breeze-tts, which also
makes the TTS output rate a per-descriptor value stated by READY instead of a
protocol-wide constant. qwen3-tts-streaming (a fourth TTS model_task, Qwen3-TTS 1.7B
Base, 24 kHz output) uses the same descriptor fields and pins no options.

## Purpose

VoiceManager runs its streaming STT engines in-process, but the audio.cpp runtime
is a native CUDA library whose fatal faults (a GGML_ASSERT on a CUDA alloc, etc.)
would take the whole FastAPI server down with it. So the audio.cpp Nemotron-3.5
streaming engine runs in a separate **sidecar process**, and VoiceManager talks to
it over this socket protocol. The process boundary is the crash isolation;
this document is the contract across it.

The sidecar drives audio.cpp through its **generic** runtime interface
(`IStreamingVoiceTaskSession`: `process_audio_chunk` -> `StreamEvent`, `finalize` ->
`TaskResult`) plus the model registry, never the concrete `NemotronASRStreamingSession`.
This wire protocol is deliberately shaped on audio.cpp's own taxonomy
(`VoiceTaskKind`, `RunMode`, `StreamingInputKind`, `StreamEvent`, `TaskResult` in
`include/engine/framework/runtime/session.h`) so that adding a second model+task later
is a registry entry plus a payload, not a protocol change. The first cut ships exactly
one model+task: Nemotron-3.5 streaming ASR.

Endpointing stays on VoiceManager's side. The spike confirmed the streaming
session emits no native acoustic end-of-utterance event; finals are produced only by an
explicit `finalize()` that the driver invokes. So on this wire, VoiceManager's existing
silero-VAD + speculative endpoint-hold decides *when* to finalize and sends a FLUSH; the
sidecar never finalizes an utterance on its own.

## Transport

- **AF_UNIX SOCK_STREAM** (Unix domain socket). Local only, no TCP port, low latency;
  matches VoiceManager's "local server, not containerized" constraint. The spike
  showed per-chunk GPU compute of ~8.5-9.5 ms at 35-66x real time, so a local socket
  round-trip for a ~0.34 s PCM chunk is noise against inference.
- The **sidecar is the server**: it binds and listens on a socket path given by
  `--socket <path>`. VoiceManager connects as the client.
- **One connection is one streaming session.** This mirrors VoiceManager's per-connection
  pipeline model: each `/ws/stream` connection that selects this engine opens its own
  sidecar connection. Closing the socket ends the session.
- **One live session per sidecar PROCESS.** A sidecar serves a single session at a time:
  it admits a connection under a process-wide lock and refuses a concurrent second one
  immediately with ERROR `busy` (then closes it). This is deliberate, not a queueing
  limitation - the process boundary is what contains a process-fatal audio.cpp fault, and
  a fault kills every session sharing the process, so a fault leaves concurrent sessions
  unaffected only while one process carries one session. **Clients get concurrency
  by running a sidecar per session, not by multiplexing one.** VoiceManager's default does
  exactly that: every `/ws/stream` session spawns its own sidecar on a private socket path
  (`VM_AUDIOCPP_SOCKET` blank). Pointing several consumers at one shared `--socket` sidecar
  is a dev/measurement arrangement, and the second concurrent one will be told `busy`.
- The model is loaded on a valid handshake and held warm across chunks and across
  utterances on the same connection (the measured warm latency presupposes this); a
  READY ack means the session is already loaded and warm. The first cut loads per
  connection - caching a loaded model across connections is a later optimisation.

## Framing

Every message is a length-prefixed frame:

```
+--------+------------------+-----------------------------+
| type   | length           | payload                     |
| u8     | u32 big-endian   | `length` bytes              |
+--------+------------------+-----------------------------+
```

- 5-byte fixed header: 1 byte message `type` tag, then a 4-byte big-endian
  (network-order) unsigned payload length.
- The header is followed by exactly `length` bytes of payload. `length` may be 0.
- Big-endian length keeps the framing trivial to implement in any language.
- Implementations MUST reject a frame whose declared `length` exceeds
  **16 MiB** (`0x0100_0000`) as a malformed frame and close the connection.

Two payload encodings, selected by the message type:

- **Control frames** carry a UTF-8 JSON object as the payload.
- **Audio frames** carry raw **little-endian signed 16-bit PCM, mono, 16 kHz**
  (`s16le`) as the payload - byte-for-byte the same audio format VoiceManager already
  validates on the first frame (`AUDIO_FORMAT_SPEC`), so no re-encode happens at the
  boundary.
- **TTS audio frames** (TTS_AUDIO, S->C) are the same raw `s16le` mono encoding at
  **the selected model_task's output rate**, stated by the READY ack of a TTS handshake:
  48 kHz for dots-tts-streaming and voxcpm2-tts-streaming (their vocoders' native
  rate), 24 kHz for breeze-tts-streaming and qwen3-tts-streaming. The client takes the
  rate from READY and
  never assumes 16 kHz - or any other fixed rate - for synthesis output.

## Message types

| tag    | name  | dir   | payload | status |
|--------|-------|-------|---------|--------|
| `0x01` | HELLO | C->S  | JSON    | v1 |
| `0x02` | READY | S->C  | JSON    | v1 |
| `0x03` | AUDIO | C->S  | s16le PCM | v1 |
| `0x04` | EVENT | S->C  | JSON    | v1 |
| `0x05` | FLUSH | C->S  | JSON    | v1 |
| `0x06` | FINAL | S->C  | JSON    | v1 |
| `0x07` | RESET | C->S  | JSON    | v1 |
| `0x08` | ERROR | S->C  | JSON    | v1 |
| `0x09` | BYE   | C->S  | JSON    | v1 |
| `0x0A` | TEXT      | C->S  | JSON    | v1 (tts) |
| `0x0B` | TTS_AUDIO | S->C  | s16le PCM at the READY rate | v1 (tts) |
| `0x0C` | TTS_DONE  | S->C  | JSON    | v1 (tts) |
| `0x0D` | CANCEL    | C->S  | JSON    | v1 (tts) |

"C->S" is client (VoiceManager) to sidecar; "S->C" is the reverse.

**Which frame set applies is implied by the HELLO's `model_task`.** An ASR model_task
(`task=asr`) speaks the 0x03-0x07 streaming-ASR set exactly as before - that set is
byte-identical to what shipped pre-TTS. A TTS model_task (`task=tts`) speaks the
0x0A-0x0D set instead: the client sends TEXT requests and receives TTS_AUDIO frames
followed by exactly one TTS_DONE per request, and may CANCEL the in-flight request.
HELLO, READY, ERROR, and BYE are common to both. The addition is not a version bump:
protocol stays 1, and an ASR-only client never sees a TTS frame.

### HELLO (0x01, C->S)

The first frame the client sends on a new connection. Carries the wire model/task
identifier.

```json
{ "protocol": 1, "model_task": "nemotron-3.5-asr-streaming" }
```

- `protocol` (int, required): wire protocol version. Must equal the sidecar's version
  (currently 1) or the sidecar replies ERROR `unsupported_protocol` and closes.
- `model_task` (string, required): the registry key selecting the model+task. Must be a
  registered id (see Model/task registry) or the sidecar replies ERROR
  `unknown_model_task` and closes.

### READY (0x02, S->C)

The handshake ack, sent in reply to a valid HELLO. Echoes the resolved task and states
the audio contract so the client needs no out-of-band knowledge.

```json
{
  "ok": true,
  "protocol": 1,
  "model_task": "nemotron-3.5-asr-streaming",
  "family": "nemotron_asr",
  "task": "asr",
  "mode": "streaming",
  "input": "audio_chunks",
  "audio": { "sample_rate": 16000, "channels": 1, "format": "s16le" },
  "preferred_chunk_samples": 512
}
```

- `task`, `mode`, `input` are the lowercase spellings of audio.cpp's
  `VoiceTaskKind::Asr`, `RunMode::Streaming`, and `StreamingInputKind::AudioChunks`.
- `preferred_chunk_samples` mirrors `StreamingPolicy::preferred_audio_chunk_samples`
  (the streaming session default is 512). The client MAY send AUDIO frames of any
  length; this is a hint.

For a **TTS model_task** the ack instead describes the output-audio contract -
`task` is `tts`, `input` is `none` (the request text arrives in TEXT frames, there is
no streaming input feed), and `audio` states what TTS_AUDIO payloads carry:

```json
{
  "ok": true,
  "protocol": 1,
  "model_task": "dots-tts-streaming",
  "family": "dots_tts",
  "task": "tts",
  "mode": "streaming",
  "input": "none",
  "audio": { "sample_rate": 48000, "channels": 1, "format": "s16le" }
}
```

- `audio` here is the **output** contract: every TTS_AUDIO payload on this connection
  is raw s16le mono PCM at exactly this rate. The rate is the descriptor's, per
  model_task: 48000 for dots-tts-streaming and voxcpm2-tts-streaming, 24000 for
  breeze-tts-streaming and qwen3-tts-streaming. The client MUST take the rate from
  READY, never assume the
  ASR 16 kHz or a fixed TTS rate.
- `preferred_chunk_samples` is an input-side hint and is absent for a TTS task.

### AUDIO (0x03, C->S)

One streaming input chunk: the payload is raw `s16le` PCM (no JSON wrapper). This is
`StreamingInputKind::AudioChunks`. Each AUDIO frame is fed to
`session.process_audio_chunk(chunk)`; the sidecar decodes it to float samples in
`[-1, 1)` (the AudioBuffer convention). Any `StreamEvent` the call returns is forwarded
as an EVENT. Chunk size is the client's choice; ~0.34 s chunks matched the spike.

### EVENT (0x04, S->C)

An interim streaming result, shaped on `StreamEvent`. `is_final` carries the MODEL's
flag honestly: false on growing-transcript deltas, true on a text-carrying
final event. A marker-only final event (is_final set, no text content) is suppressed -
the FINAL frame from FLUSH remains the single utterance-final.

The wire permits EVENTs at any point after READY. For the Nemotron-3.5 ASR streaming
model, `process_audio_chunk` encodes each chunk as it arrives and emits the growing
deltas - via the session's stream-event sink - during speech; `finalize()` then decodes
only the tail plus a silence flush, so a few more EVENTs may precede the FINAL. The
joined deltas usually match the FINAL text, but a decode may revise text it already
sent, so the FINAL is the authoritative text. A model that returns
partials directly from `process_audio_chunk` (in its `StreamEvent`) is handled the same
way; the drive loop forwards sink and returned events without change.

```json
{
  "is_final": false,
  "partial_text": { "text": "long range scan", "language": "en" },
  "voice_activity": [],
  "word_timestamps": []
}
```

- `partial_text` maps to `StreamEvent::partial_text` (a `Transcript`). audio.cpp emits
  `partial_text` as token/char deltas; the sidecar MAY forward deltas or an accumulated
  string - the field is defined as the current best partial transcript for the utterance.
- `voice_activity` and `word_timestamps` map to the same-named `StreamEvent` fields and
  are empty unless the session populates them (the spike saw no `voice_activity` events).

### FLUSH (0x05, C->S)

Asks the sidecar to finalize the current utterance. VoiceManager's endpointer sends this
at the endpoint-hold expiry (the finalize decision stays ours). The sidecar calls
`session.finalize()` and replies with exactly one FINAL. Payload: `{}`.

### FINAL (0x06, S->C)

The utterance-final result, shaped on `TaskResult`, sent in reply to FLUSH.

```json
{
  "is_final": true,
  "text_output": { "text": "long range scan of planet complete", "language": "en" },
  "word_timestamps": []
}
```

- `text_output` maps to `TaskResult::text_output` (a `Transcript`); `word_timestamps`
  maps to `TaskResult::word_timestamps`.
- Exactly one FINAL is sent per FLUSH. After a FINAL the session is ready for the next
  utterance (the client sends RESET to start a fresh one on the same connection).

### RESET (0x07, C->S)

Resets the streaming session state for a new utterance without reloading the model
(`session.reset()`), keeping it warm. Payload: `{}`. No reply.

### ERROR (0x08, S->C)

A protocol or runtime error. The sidecar sends ERROR and then, for a handshake-level
error, closes the connection.

```json
{ "ok": false, "code": "unknown_model_task", "message": "no such model_task; known: [\"nemotron-3.5-asr-streaming\", \"voxtral-mini-4b-realtime-asr-streaming\", \"dots-tts-streaming\", \"voxcpm2-tts-streaming\", \"breeze-tts-streaming\", \"qwen3-tts-streaming\"]" }
```

Defined `code` values:

- `unsupported_protocol` - HELLO `protocol` does not match the sidecar's version.
- `unknown_model_task` - HELLO `model_task` is not a registered id.
- `bad_handshake` - the first frame was not a well-formed HELLO.
- `busy` - the sidecar already has a live session (one per process, see Transport). Sent
  before the HELLO is even read, then the connection closes. A client that wants another
  concurrent session must start its own sidecar rather than retry against this one.
- `bad_frame` - a frame was malformed (bad JSON, oversized length, or an unexpected type).
- `not_implemented` - a spec'd-but-not-yet-built message type.
- `internal` - an unexpected sidecar-side error.
- `prepare_failed` - the TTS session's prepare step failed after READY (an engine-side
  load/validate error). Sent instead of a bare EOF; the connection then closes.
- `sidecar_fault` - a **process-fatal** fault (best-effort; see below). Unlike the others
  it is not followed by a graceful close: the sidecar process dies immediately after.

### Process-fatal faults

The audio.cpp runtime is native CUDA code whose fatal faults (a `GGML_ASSERT` -> `abort()`
on a CUDA alloc, a `SIGSEGV`, ...) cannot be caught and recovered - the whole sidecar
process dies. The **process boundary is the crash isolation**: such a fault stays in the
sidecar and never reaches the VoiceManager server.

On the way down, the sidecar makes one **best-effort**, async-signal-safe attempt to send an
ERROR `sidecar_fault` frame to the connected client, then re-raises the signal so it exits
with the real fault's status. The client is therefore guaranteed a **defined disconnect
within a bound** - an ERROR `sidecar_fault` and/or a socket EOF - never a hang. Because the
frame is best-effort (e.g. a fault mid-write, or an uncatchable `SIGKILL`), the client MUST
treat an **unexpected socket EOF the same as a fault**: fail the affected session with a
defined error and, if desired, respawn the sidecar. A freshly spawned sidecar reclaims a
stale socket file (it `unlink`s the path before binding), so a restart accepts a new session.

### BYE (0x09, C->S)

Graceful close request. The sidecar finishes any in-flight reply, then closes the
connection and frees the session. Payload: `{}`. A plain socket EOF is treated the same
way.

### TEXT (0x0A, C->S)

A synthesis request, valid only on a connection whose HELLO selected a TTS model_task.
Carries the text to speak plus the voice reference material. `ref_text` is **per-model**:
required and non-empty for `dots-tts-streaming` (dots_tts conditions on the
reference WAV *and* its exact transcript), optional for `voxcpm2-tts-streaming`
(voxcpm2 ignores the transcript unless continuation audio is also supplied). The
sidecar enforces this via the descriptor's `ref_text_required` flag, never a family
string comparison: a dots TEXT frame missing an empty `ref_text` is rejected with
ERROR `bad_frame` before dispatch; a voxcpm2 TEXT frame may omit `ref_text` and the
reference_transcript option key is then simply not set on the request.

```json
{
  "text": "Warp core is stable.",
  "voice_wav": "/abs/path/to/designed_female_s101.wav",
  "ref_text": "the exact transcript of that reference wav"
}
```

- `text` (string, required): the text to synthesize.
- `voice_wav` (string, required): absolute path to the reference-speaker WAV, readable
  by the sidecar process (same machine - this is an AF_UNIX-local protocol).
- `ref_text` (string, required for dots, optional for voxcpm2): the exact transcript of
  `voice_wav`. A voxcpm2 request may omit it (or send it empty).

One TEXT frame is one synthesis request; the reply is a stream of TTS_AUDIO frames
closed by exactly one TTS_DONE. The registry's request-option defaults (for dots:
`text_chunk_size=200`, `max_tokens=500`; for voxcpm2: `retry_badcase=false`, mandatory
for streaming - generate_streaming throws without it - and `voxcpm2.chunk_strategy=stateless`,
which keeps each long-form chunk synthesised from the reference prompt alone as release-0.6
did, where v0.7.3 defaults to carrying the previous chunk forward) apply to every request; the
engine chunks long text internally, the client does not pre-chunk.

### TTS_AUDIO (0x0B, S->C)

One chunk of synthesized audio: the payload is raw `s16le` mono PCM at the rate the
READY ack stated (48000 for dots and voxcpm2, 24000 for breeze and qwen3) - no JSON
wrapper, no
re-encode. Chunks arrive
in generation order as the model produces them (streaming, not one blob at the end);
chunk size is the sidecar's choice and carries no meaning. The client just
concatenates payloads in arrival order.

### TTS_DONE (0x0C, S->C)

The end-of-request marker, sent only after the last TTS_AUDIO byte of the request it
closes - when the client reads TTS_DONE it has every audio byte.

```json
{ "ok": true }
```

A request ended by a CANCEL carries the marker instead:

```json
{ "ok": true, "cancelled": true }
```

After TTS_DONE the connection is ready for the next TEXT request. A request that fails
mid-generation is closed by ERROR instead; TTS_DONE without `cancelled` always means a
complete, untruncated synthesis, and with it a cleanly-stopped one.

### CANCEL (0x0D, C->S)

Stop the in-flight TTS request. Payload is an empty JSON object:

```json
{}
```

Sent while a TEXT request is streaming, it stops generation promptly (the drive
path's sink throws a sentinel between vocoder merge groups, so the worst-case stop
latency is one merge group - about 0.35 s of audio at the pinned dots defaults).
Audio frames already written when the CANCEL lands may still arrive; the sidecar
discards its own queued-but-unwritten frames, and the request is then closed by
`TTS_DONE {"ok":true,"cancelled":true}`. The session stays warm - the same
connection serves the next TEXT request normally (sink-throw session reuse, the
verdict of the sink-throw cancel probe).

A CANCEL that arrives with no request in flight (the client cancelled just as the
request finished - a benign race) is ignored: no ERROR, no reply. The client should
treat the request's already-sent TTS_DONE as its terminal either way.

## Model/task registry

The sidecar holds a small registry mapping each wire `model_task` id to a task
descriptor. Six entries ship (the second added by the audiocpp-second-model project;
the third by audiocpp-tts-backend; the fourth by audiocpp-voxcpm2-tts;
the fifth by audiocpp-breeze-tts, served since the audio.cpp v0.7.3 pin
by the sidecar build on the current pin; the sixth, qwen3-tts-streaming, served since
main 624cf918):

| wire id                                   | family             | task | mode      | input        | output rate |
|-------------------------------------------|--------------------|------|-----------|--------------|-------------|
| `nemotron-3.5-asr-streaming`              | `nemotron_asr`     | asr  | streaming | audio_chunks | -           |
| `voxtral-mini-4b-realtime-asr-streaming`  | `voxtral_realtime` | asr  | streaming | audio_chunks | -           |
| `dots-tts-streaming`                      | `dots_tts`         | tts  | streaming | none         | 48000       |
| `voxcpm2-tts-streaming`                   | `voxcpm2`          | tts  | streaming | none         | 48000       |
| `breeze-tts-streaming`                    | `breeze_tts`       | tts  | streaming | none         | 24000       |
| `qwen3-tts-streaming`                     | `qwen3_tts`        | tts  | streaming | none         | 24000       |

The TTS descriptors carry their output `sample_rate` (48000 is the default; breeze
and qwen3 override it), and READY advertises that value - the rate is per model_task, not a
protocol constant.

`family`/`task`/`mode` are what the drive loop passes to audio.cpp's generic
registry: `registry.load({model_path, family_hint = family})` then
`model->create_task_session({VoiceTaskKind, RunMode}, options)`, cast to the streaming
session interface. No concrete session class name appears on this wire or in the drive
loop, and the drive path is selected by the descriptor's `task` kind, never the model
family. The nemotron entry carries no default options (shipped behavior is exactly the
engine defaults); the voxtral entry pins its arena/context recipe as session-option
defaults; the dots entry pins the upstream long-form guards (`text_chunk_size=200`,
`max_tokens=500`) as request-option defaults and requires `ref_text`; the voxcpm2
entry pins `retry_badcase=false` (streaming throws without it) and
`voxcpm2.chunk_strategy=stateless`, and makes `ref_text` optional; the breeze entry pins `text_chunk_size=200` and
`guidance_scale=1.0` as its only request-option defaults, requires `ref_text`, and
declares 24000 as its output rate; the qwen3 entry pins no options, requires `ref_text`
and declares 24000 - all carried on the descriptor
(`ref_text_required`, `sample_rate`), never a family string comparison in the drive
loop. The drive path also reads the session's declared streaming output kind (push
through the sink, or pull events one chunk at a time as breeze does) from the engine's
streaming policy, again never from the family. Adding another model+task is one more
registry entry plus its payload; the drive loop does not change.

## A session, end to end (informative)

```
client                         sidecar
  |----- HELLO {proto,id} ------->|   validate proto + registry id
  |<---- READY {audio,...} -------|   session held warm
  |                               |
  |----- AUDIO (s16le pcm) ------>|   process_audio_chunk -> StreamEvent
  |<---- EVENT {partial_text} ----|
  |----- AUDIO (s16le pcm) ------>|
  |<---- EVENT {partial_text} ----|   ... growing interims
  |----- FLUSH {} --------------->|   finalize()  (our endpointer decided)
  |<---- FINAL {text_output} -----|   exactly one final
  |----- RESET {} --------------->|   reset() for the next utterance
  |             ...               |
  |----- BYE {} ----------------->|   close, free session
```

And a TTS session (a HELLO selecting `dots-tts-streaming`):

```
client                         sidecar
  |----- HELLO {proto,id} ------->|   validate proto + registry id
  |<---- READY {task:tts,rate} ---|   session held warm (rate per model_task)
  |                               |
  |----- TEXT {text,wav,ref} ---->|   synthesis request
  |<---- TTS_AUDIO (pcm@rate) -----|   streamed as generated
  |<---- TTS_AUDIO (pcm@rate) -----|   ...
  |<---- TTS_DONE {ok} -----------|   after the last audio byte
  |             ...               |   next TEXT request
  |----- TEXT {text,wav,ref} ---->|   a request the client changes its mind on
  |<---- TTS_AUDIO (pcm@rate) -----|   streaming...
  |----- CANCEL {} -------------->|   stop it
  |<---- TTS_DONE {ok,cancelled} -|   closed early; session stays warm
  |             ...               |   next TEXT request
  |----- BYE {} ----------------->|   close, free session
```

## Versioning

The single `protocol` integer in HELLO/READY governs compatibility. A breaking change
bumps it; the sidecar rejects a mismatched HELLO with ERROR `unsupported_protocol`. New
optional JSON fields that a reader can ignore are not breaking and do not bump it.
