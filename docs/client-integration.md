# VoiceManager Client Integration Reference

A single, self-contained reference for integrating a client against the
VoiceManager streaming STT/TTS service over WebSockets (plus its REST surface).
The WebSocket protocol is the real contract; the Python `client.py` is one
implementation of it. Every symbol named here exists in current source - see
[Verifying against source](#verifying-against-source).

## Table of contents

- [Quickstart](#quickstart)
- [Authentication](#authentication)
- [Integration walkthrough](#integration-walkthrough)
- [Speech-to-text (STT)](#speech-to-text-stt)
- [Text-to-speech (TTS)](#text-to-speech-tts)
- [Voice selection](#voice-selection)
- [The endpoint-hold consumption model](#the-endpoint-hold-consumption-model)
- [Turn-taking protocol surface](#turn-taking-protocol-surface)
- [WebSocket message-schema reference](#websocket-message-schema-reference)
- [Client method reference](#client-method-reference)
- [Close codes](#close-codes)
- [Configuration](#configuration)
- [Voice cloning](#voice-cloning)
- [REST endpoints](#rest-endpoints)
- [Gotchas](#gotchas)
- [Guarantees you can depend on](#guarantees-you-can-depend-on)
- [Verifying against source](#verifying-against-source)

## Quickstart

Fill in `TOKEN` and `HOST` and run it - nothing else to edit. It connects,
synthesizes one utterance with `synthesize_tts` and saves it, then feeds that
audio back through streaming STT and prints the transcript: one TTS round-trip
and one STT round-trip, end to end.

```python
"""VoiceManager Quickstart - connect, synthesize speech, transcribe it back.

Fill in TOKEN and HOST, then run:  python quickstart.py
Needs the VoiceManager client (copy client.py into your project, or install the
voicemanager package) plus its deps: websockets, httpx.
"""
import asyncio
import audioop  # stdlib (Python 3.12); resamples the TTS output for the STT round-trip

from voicemanager.client import VoiceManagerClient  # or: from client import VoiceManagerClient

TOKEN = "vmp_..."      # <-- from: voicemanager user token add USER NAME
HOST = "localhost"     # <-- your server host

STT_RATE = 16000  # the server expects 16 kHz mono 16-bit PCM for STT


async def main() -> None:
    async with VoiceManagerClient(host=HOST, token=TOKEN) as client:
        print(f"connected: session={client.session_id} "
              f"stt={client.stt_engine} tts={client.tts_engine}")

        # --- TTS round-trip: synthesize one utterance and save it -------------
        text = "Hello from VoiceManager. This is a streaming speech test."
        pcm, tts_rate = await client.synthesize_tts(text)
        client.save_tts_to_file(pcm, "quickstart_tts.wav", tts_rate)
        print(f"TTS: {len(pcm)} bytes at {tts_rate} Hz -> quickstart_tts.wav")

        # --- STT round-trip: feed that audio back and read the transcript -----
        # Resample the TTS PCM (e.g. 24 kHz) down to the 16 kHz the STT path wants,
        # then pad with a little silence so the endpointer commits a final.
        pcm16, _ = audioop.ratecv(pcm, 2, 1, tts_rate, STT_RATE, None)
        pcm16 += b"\x00\x00" * STT_RATE  # ~1s of trailing silence

        async def chunks():
            step = STT_RATE  # ~0.5s of audio per send (STT_RATE bytes = STT_RATE/2 samples)
            for i in range(0, len(pcm16), step):
                yield pcm16[i:i + step]

        results = await client.stream_stt(chunks())
        finals = [r["text"] for r in results if r.get("is_final")]
        print(f"STT: {len(results)} frame(s); final text: {finals!r}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run verbatim against a VibeVoice server (`VM_TTS_ENGINE=vibevoice`), this prints
the following - the exact transcript wording varies with the STT engine:

```text
connected: session=cb134688-247b-4300-a057-ac0114fd0cd3 stt=nemotron tts=vibevoice
TTS: 172800 bytes at 24000 Hz -> quickstart_tts.wav
STT: 4 frame(s); final text: [' Hello from Voice Manager, this is uh streaming speech test']
```

`audioop` is used only to downsample the demo's own audio for the STT leg; it is
in the standard library through Python 3.12 (removed in 3.13). In real use you
would stream microphone or file audio that is already 16 kHz mono 16-bit PCM (see
[transcribe_file](#client-method-reference)), so no resampling is needed.

## Authentication

Every client authenticates with a personal bearer token. The server operator
creates a user and a token with the CLI on the server host:

```sh
voicemanager user add alice
voicemanager user token add alice hermes-console
```

`user token add` prints the secret once: `vmp_` followed by 43 URL-safe
characters. Store it; the server keeps only its hash and cannot show it again.
A signed-in user can also create and revoke their own tokens, and set a token's
voice, on the web UI at `http://<host>:<port>/tokens`.

Send the token in the `Authorization` header, on the WebSocket upgrade request
and on every authenticated REST call:

```text
GET /ws/stream HTTP/1.1
Host: voice.example:8000
Upgrade: websocket
Connection: Upgrade
Authorization: Bearer vmp_...
```

The token never goes in a URL; a token in a query parameter is ignored. Any
WebSocket library that can set a header on the upgrade request works (Python
`websockets`, Kotlin OkHttp). Browser WebSocket clients cannot set the header
and are not supported.

| Where | No token or an invalid one | What the client sees |
|---|---|---|
| `/ws/stream` upgrade | rejected before accept | HTTP `403` on the upgrade request; no WebSocket and no `ready` frame. |
| `GET /info`, `POST /voice/upload` | refused | HTTP `401` with `WWW-Authenticate: Bearer` and body `{"detail": "missing bearer token"}` or `{"detail": "invalid token"}`. |
| `GET /health` | open | No token needed. |

An invalid token is an unknown token, a revoked token, or a header with a scheme
other than `Bearer`. A revoked token is refused on the next connect; a WebSocket
that is already open stays open.

A session belongs to the user of the token that opened it. `/info` lists only
your own sessions, and `POST /voice/upload` for another user's session answers
`404`, the same as an unknown session id.

The Python client does all of this for you: `VoiceManagerClient(token=...)`
sends the header and raises `AuthenticationError` on a `401` or `403`.

> **Removed: the shared key.** The `api_key` query parameter and the server's
> `VM_API_KEY` setting no longer exist. A server that still finds `VM_API_KEY` in
> its environment or `.env` refuses to start and says so: remove it and create
> tokens with `voicemanager user token add`. A client that still sends only the
> old query parameter gets a `403` at the WebSocket upgrade and a `401` from REST.

## Integration walkthrough

The [Quickstart](#quickstart) is one runnable file; this section is the ordered
tour behind it. It walks the full lifecycle of a connection - connect, handshake,
stream STT, consume results, synthesize TTS, cancel, tear down - one step at a
time, linking to the reference sections for the exhaustive field-by-field detail.
The snippets assume the reference Python client and an open `async with` scope;
any language can follow the same order over the raw WebSocket contract.

### 1. Connect and authenticate

Construct a `VoiceManagerClient` with the server host, port, and your token,
then open the connection by entering the async context manager. No socket opens
until you enter the context; on entry the client dials `/ws/stream` with the
token in the `Authorization: Bearer` header (see [Authentication](#authentication)).

```python
from voicemanager.client import VoiceManagerClient, AuthenticationError

client = VoiceManagerClient(host="localhost", port=8000, token="vmp_...")
try:
    async with client:
        ...  # steps 2-7 happen inside this scope
except AuthenticationError:
    ...  # token refused - do not retry with the same one
```

Auth is verified **before** the server accepts the WebSocket upgrade: a bad token
is an HTTP `403` at the handshake, not a close frame, and the client surfaces it
as `AuthenticationError` rather than a `1008` close event. See
[Close codes](#close-codes) for why `1008` can never actually reach a client.

### 2. Read the handshake

The first frame the server sends is `ready` (`HandshakeMessage`). Entering the
context manager reads it for you and populates the client's public attributes
before the `async with` body runs, so the session id and active engines are
available immediately:

```python
async with client:
    print(client.session_id)       # e.g. "cb134688-247b-4300-a057-ac0114fd0cd3"
    print(client.stt_engine)       # e.g. "nemotron"
    print(client.tts_engine)       # e.g. "vibevoice"
    print(client.available_voices) # e.g. ["Carter", "Emma", ...], or [] under OpenVoice
    print(client.default_voice)    # the voice a request without voice gets, or None
```

`session_id` is the handle you pass to `POST /voice/upload` for
[voice cloning](#voice-cloning); the engine fields tell you which behaviours are
live (only `nemotron` drives the [endpoint-hold path](#the-endpoint-hold-consumption-model);
VibeVoice and audiocpp list their presets in `available_voices`, and
`default_voice` follows the [voice selection](#voice-selection) order). Full field list:
[`ready` - `HandshakeMessage`](#ready---handshakemessage).

### 3. Stream speech to text

Feed the server 16 kHz mono 16-bit PCM as binary frames; it streams back
`transcription` frames as dicts. Use `stream_stt_live` to react to each result
the moment it arrives (both interim and final), or `stream_stt` to collect them
all and return a list. The first binary frame is format-validated - a mismatch
closes the socket with `1003` (see [Binary frames](#binary-frames)).

```python
async def mic_frames():
    # yield raw 16 kHz mono 16-bit PCM chunks (e.g. ~0.5 s each) from mic or file
    ...

async for frame in client.stream_stt_live(mic_frames()):
    print(frame["segment_index"], frame["is_final"], frame["text"])
```

See [Speech-to-text (STT)](#speech-to-text-stt) and the
[method reference](#client-method-reference).

### 4. Consume endpoint vs is_final

Each `transcription` dict carries a `segment_index`, an `endpoint` flag, and an
`is_final` flag. Treat `segment_index` as a **stable key**, not an append cursor:
keep a map from index to its latest text and overwrite the slot on every frame
for that index (upsert-by-index). Act irreversibly only on `is_final=true`; treat
`endpoint=true` as a speculative low-latency hint you may act on early, knowingly
accepting that it can still be revised.

```python
segments: dict[int, str] = {}
async for frame in client.stream_stt_live(mic_frames()):
    segments[frame["segment_index"]] = frame["text"]   # upsert, never append
    if frame["is_final"]:
        dispatch(frame["text"])                         # commit - safe to act
    elif frame["endpoint"]:
        prefetch(frame["text"])                         # speculative - may be revised
```

This is the crux of consuming the streaming path correctly; the full mental model
(the three frame kinds, the revision contract, and the `endpoint` vs `is_final`
distinction) is in [The endpoint-hold consumption model](#the-endpoint-hold-consumption-model).

### 5. Synthesize speech (TTS)

There are two shapes. **One-shot** (`synthesize_tts`) takes a complete string and
returns all the audio at once - use it when you already have the full text:

```python
pcm, rate = await client.synthesize_tts("Your flight is booked.", voice="Carter")
client.save_tts_to_file(pcm, "reply.wav", rate)
```

**Streaming** (`stream_tts_live`) takes an async iterator of text fragments (e.g.
an LLM token stream) and yields PCM as it is synthesized - use it for low-latency
playback while the text is still being produced:

```python
async def token_stream():
    for tok in ["Your ", "flight ", "is ", "booked."]:
        yield tok

async for pcm_chunk in client.stream_tts_live(token_stream(), voice="Carter"):
    play(pcm_chunk)   # sample rate available via client.last_tts_sample_rate
```

**Per-segment seam.** One-shot synthesis renders the whole string as a single
utterance with continuous prosody. Streaming synthesizes each speakable unit the
server accumulates as its own segment, and VibeVoice **restarts its prosody per
segment** - so splitting one sentence across several fragments can seat an audible
seam at each segment boundary. Send whole speakable units (a full clause or
sentence) per synthesized segment, and flush only at a boundary you have chosen;
do not split mid-word. (The audiocpp engine softens this: it coalesces
busy-period text into one request, so seams land only at request boundaries -
see [its section](#the-audiocpp-tts-engine-presets-per-model-sample-rate-one-request-batching).)
See [Text-to-speech (TTS)](#text-to-speech-tts) for the
request/response frames and the `flush`/`done` control semantics.

### 6. Cancel an in-flight reply

To stop a TTS reply mid-stream (barge-in), call `cancel_tts` with its
`request_id`. Supply the id up front to `stream_tts_live`, or read
`client.last_tts_request_id` once iteration has started, then cancel from a
separate task:

```python
import uuid, asyncio

rid = str(uuid.uuid4())
player = asyncio.create_task(play_stream(client.stream_tts_live(token_stream(), request_id=rid)))
# ... on a barge-in signal:
await client.cancel_tts(rid)   # fire-and-forget; server acks with an immediate audio_end
```

`cancel_tts` returns as soon as the `tts_cancel` frame is sent. The server acks
with exactly one `audio_end` carrying `reason: "cancelled"`, and the client
discards any PCM still in flight for that id until the ack lands. A cancelled
`request_id` is permanently dead - **mint a fresh id for the next reply**. See
behaviours 5 and 6 in [Turn-taking protocol surface](#turn-taking-protocol-surface).

### 7. Tear down

Leaving the `async with` scope closes the WebSocket, cancels the single reader
task, and drains the dispatch queues - no explicit disconnect call is needed.
Everything the connection held is **volatile**: the session id, any uploaded
voice profile, and all turn/segment state are in-memory and gone once the socket
closes.

```python
async with client:
    ...  # steps 2-6
# socket closed here; client.session_id and any voice profile are gone
```

If the client transparently reconnects mid-session it sets `reconnected = True`
and starts a fresh server-side session, so a previously uploaded voice profile
must be re-uploaded (it clears the flag once you do). For a clean end of work,
simply exit the context manager.

## Speech-to-text (STT)

STT streams over the same socket: send 16 kHz mono 16-bit PCM as binary frames
and receive `transcription` frames. The
[endpoint-hold consumption model](#the-endpoint-hold-consumption-model) explains
how interim, endpoint, and final frames revise one utterance; the
[message-schema reference](#websocket-message-schema-reference) has the frame
fields; walkthrough [step 3](#3-stream-speech-to-text) is a runnable example.

### The audio.cpp STT engine: cadence depends on the served model_task

When the server runs `VM_STT_ENGINE=audiocpp`, the STT runtime is an audio.cpp
sidecar process serving ONE registered model+task, selected server-side by
`VM_AUDIOCPP_MODEL_TASK`. Two are registered:

- `nemotron-3.5-asr-streaming` (the default) - Nemotron-3.5, mid-speech
  streaming (growing partials while the speaker talks, a few words at a time);
- `voxtral-mini-4b-realtime-asr-streaming` - Voxtral-Mini-4B-Realtime, genuine
  mid-speech streaming (growing partials while the speaker talks).

`VM_AUDIOCPP_MODEL` must point at the matching GGUF file. An unknown model_task
id is refused at session start with cause code `unknown_model_task` naming the
valid ids (a permanent config failure: close `4001`). Per-model tuning ships as
registry defaults inside the sidecar; operators can override or extend them with
`VM_AUDIOCPP_SESSION_OPTIONS` / `VM_AUDIOCPP_REQUEST_OPTIONS` (comma-separated
KEY=VALUE lists, forwarded opaquely).

Check `client.stt_engine == "audiocpp"` on the handshake to branch behaviour.
The handshake does not reveal WHICH model_task the server is configured with, so
render defensively: treat `segment_index` as an upsert key and replace the
displayed text with each `is_final=false` frame (that model works unchanged for
both cadences below). On either model_task there is no `endpoint=true` frame
(that is the `nemotron`-engine-only endpoint-hold path).

#### With the nemotron model_task (the default): growing partials during speech

- **Live mid-speech partials, replace semantics.** The model encodes each audio
  chunk as it arrives, so while you speak the server emits `is_final=false`
  partials, each carrying the CUMULATIVE transcript-so-far for the current
  utterance - never a lone word or delta. Always REPLACE the rendered text for
  that `segment_index` with the newest frame; never append frames together. After
  the pause, the server decodes the tail of the utterance and exactly one
  `is_final=true` final commits the complete text (the last word or so may first
  appear in the final); the next utterance starts a fresh transcript at the next
  `segment_index`. Measured on this rig on a warm connection, the first partial
  lands ~1.3-2.3 s after speech onset and then grows a few words at a time while
  the speaker talks. The server forwards partials as your audio frames arrive, so
  the exact timing also depends on your frame size (those figures used 320 ms frames).

#### With the voxtral model_task: growing partials during speech

- **Live mid-speech partials, replace semantics.** While you speak, the server
  emits a stream of `is_final=false` partials, each carrying the CUMULATIVE
  transcript-so-far for the current utterance - never a lone word or delta.
  Always REPLACE the rendered text for that `segment_index` with the newest
  frame; never append frames together. At end-of-utterance exactly one
  `is_final=true` final commits the utterance (same text contract), and the next
  utterance starts a fresh transcript at the next `segment_index`. Measured on
  this rig the first partial lands ~1.3-2.6 s after speech onset and then grows
  word-by-word (~80 ms token cadence).
- **Very short utterances yield NO frames at all.** Under roughly 0.6 s of
  speech the model produces no output: no partial, no empty final - the same
  outcome as the VAD gating other engines apply to non-speech. The session stays
  healthy and the next utterance transcribes normally. Do not treat a silent
  gap in frames as an error; `segment_index` may legitimately skip values
  consumed by such empty utterances.
- **Per-session VRAM cost and ceiling.** A voxtral session holds ~3.6 GiB of
  VRAM (3,579 MiB measured, stream-length independent), against ~1.5 GiB for
  the nemotron model_task. On the shipped 16 GiB serving device that is 4
  sessions on an empty card and an honest **3 concurrent sessions** alongside
  today's resident tenants (a 24 GiB 3090 fits 6). `VM_AUDIOCPP_MAX_SIDECARS`
  (default 4) still admission-caps sessions; over-cap connections get the same
  retriable `4002` / `at_capacity` refusal described below.
- **What ends an utterance.** The server finalizes after a gap in inbound audio -
  `VM_AUDIOCPP_SILENCE_MS` of silence (default 800 ms). Keep sending audio frames
  during speech; the pause you leave is what triggers the transcript.
- **Time to first text (nemotron model_task).** The first partial usually arrives
  during speech, ~1.3-2.3 s after speech onset (see its section above), so budget
  for "text while speaking." A short utterance (about 2 s) can end before its first
  partial. The server sends a ready partial only when your next audio frame
  arrives: a client that keeps streaming frames through the pause gets it on the
  next frame, and a client that stops sending gets it just before the final. Warm,
  the final lands about `VM_AUDIOCPP_SILENCE_MS + ~120 ms` after the speaker stops
  (e.g. ~0.9 s with the default), because the server finalizes only after the pause.
- **Warm-up: hold one long-lived connection.** Each `/ws/stream` connection gets its
  own audio.cpp sidecar process, and the first utterance on a *new* connection pays
  for standing it up: process spawn, CUDA context init, model load, then the first
  decode. Measured on this rig's production config (cuda:3, f16 model already in
  the OS page cache, Breeze TTS on the same card), that one-time cost is **~1.6 s**, so the
  final of the first utterance on a new connection lands about
  `VM_AUDIOCPP_SILENCE_MS + ~1.7 s` after the speaker stops (about 1.8 s at a
  100 ms silence window), against `+ ~120 ms` for every later utterance on that
  same connection. About 0.45 s of that one-time cost is the per-connection Breeze
  TTS warm-up the server starts on the same card at connect; with a TTS engine
  that spawns nothing per connection it is ~1.1 s (about 1.3 s at a 100 ms
  window). The voxtral q4_k model is larger: ~2.7 s page-cached. Open ONE
  `/ws/stream` connection and keep it open for the session - do not reconnect per
  utterance, or you pay that cost every time. If your first utterance's latency
  matters, send a short throwaway clip right after connect to warm the session.
  Two caveats: on the very first connection after a reboot the model is not yet page
  cached and the load is slower (it reads ~1.3 GB from disk); and if an operator runs
  an external warm sidecar (`VM_AUDIOCPP_SOCKET`) instead of the default, the spawn
  and load are already paid and the per-connection cost drops to the decode warm-up
  alone. The nemotron figures above were re-measured on the current audio.cpp pin,
  and the voxtral ~2.7 s figure was not.
- **Concurrent sessions are capped, and the cap is small.** Because each connection
  gets its own sidecar process, each concurrent session costs one process, one CUDA
  context and one resident copy of the model - all on the single GPU named by
  `VM_AUDIOCPP_DEVICE`. Measured on the shipped default (Nemotron-3.5 f16, RTX
  4060 Ti): **~1.5 GiB of VRAM per concurrent session** (1,476-1,490 MiB measured on
  the current pin), on top of the ~2.8 GiB the VoiceManager server itself holds, so
  roughly **8 sessions fit on a 16 GiB card**.
  The server admits at most `VM_AUDIOCPP_MAX_SIDECARS` (default 4) at once and
  refuses the rest with close `4002` / `stt_backend_unready` and cause code
  `at_capacity` - a *retriable* refusal, not a configuration fault. Treat it as
  backpressure: retry with backoff and it clears as soon as another session ends.
  This is a much lower concurrency ceiling than the other STT engines, which share
  one in-process model across all sessions; if you need many simultaneous
  transcription sessions, this engine is not the right choice today.

Both model_tasks stream growing partials during speech; they differ in how the
text grows (nemotron a few words per audio chunk, voxtral word-by-word at token
cadence) and in cost (see their sections above). Design consumers around the
defensive model either way - upsert by `segment_index`, replace text on every
`is_final=false` frame, commit on `is_final=true`.

## Text-to-speech (TTS)

TTS runs one-shot (`tts`) or streaming (`tts_stream`, fed token-by-token and
closed with `done`, or cut early with `flush`). Every reply is framed
`audio_start` -> PCM binary frames -> `audio_end`, the output sample rate declared
per reply on `audio_start`. Frame schemas are in the
[message-schema reference](#websocket-message-schema-reference); streaming and
flush control in [Turn-taking protocol surface](#turn-taking-protocol-surface).

### The audio.cpp TTS engine: presets, per-model sample rate, one-request batching

When the server runs `VM_TTS_ENGINE=audiocpp`, TTS is served by an audio.cpp
sidecar process - the same sidecar binary as the audio.cpp STT engine - running
the selected TTS model_task: `dots-tts-streaming` by default, or
`voxcpm2-tts-streaming` / `breeze-tts-streaming` / `qwen3-tts-streaming` by
configuration. The wire
contract is unchanged (`tts` / `tts_stream` / `tts_cancel`, framed
`audio_start` -> PCM -> `audio_end`); what changes is the output rate (declared
per model_task, see below), the voice model, and the latency/seam behavior. Check
`client.tts_engine == "audiocpp"` on the handshake to branch.

**Selection config.** `VM_TTS_ENGINE=audiocpp` selects the engine; the
`VM_AUDIOCPP_TTS_*` settings shape it (all mirrored in `.env.example`):

| Env var | Default | Meaning |
|---|---|---|
| `VM_AUDIOCPP_TTS_MODEL` | *(empty)* | Path to the TTS GGUF (dots, voxcpm2, breeze or qwen3); required to serve live. |
| `VM_AUDIOCPP_TTS_MODEL_TASK` | `dots-tts-streaming` | The registered model+task the TTS sidecar serves. `dots-tts-streaming` is the default; `voxcpm2-tts-streaming` opts into VoxCPM2 (pair with a voxcpm2 GGUF; registry pins `retry_badcase=false` and `voxcpm2.chunk_strategy=stateless`); `breeze-tts-streaming` opts into Breeze-TTS 2 (pair with a breeze GGUF; pins `text_chunk_size=200`, `guidance_scale=1.0`; 24 kHz output; served by the sidecar build on the current audio.cpp pin); `qwen3-tts-streaming` opts into Qwen3-TTS 1.7B Base (pair with the qwen3 Base GGUF; no pinned options; `ref_text` required; 24 kHz output). |
| `VM_AUDIOCPP_TTS_VOICES` | `voices/designed/refs.json` | The preset-voice manifest (see below). |
| `VM_AUDIOCPP_TTS_VOICE` | `designed_female_s101` | Default preset used when a request sends no `voice`. |
| `VM_AUDIOCPP_TTS_SEAM_SILENCE_MS` | `400` | Cap on the silence gap at an internal chunk seam (see the batching section below); `0` disables the trim. |
| `VM_AUDIOCPP_TTS_COALESCE_MS` | `350` | On `tts_stream`, how long to keep collecting sentences before synthesizing, once the first batch has dispatched (see the batching section below). The first sentence is never delayed. `0` dispatches whatever is queued at that instant. |
| `VM_AUDIOCPP_TTS_REQUEST_OPTIONS` | *(empty)* | Comma-separated `KEY=VALUE` overrides merged over the model's pinned request options (`text_chunk_size=200`, `max_tokens=500`, `text_chunk_mode=default`) - operator tuning, e.g. a larger chunk budget for fewer seams at more drift risk. |

The sidecar binary and its compute knobs are shared with the STT engine:
`VM_AUDIOCPP_SIDECAR_BIN`, `VM_AUDIOCPP_BACKEND`, `VM_AUDIOCPP_DEVICE`, and
`VM_AUDIOCPP_THREADS` apply to both. Voice assets are validated fail-fast at
startup - a manifest entry with a missing WAV (or, under dots, breeze and qwen3, an
empty reference transcript) refuses to start the server, so a running server
always has usable presets. The transcript (`ref_text`) is mandatory for
dots-tts-streaming, breeze-tts-streaming and qwen3-tts-streaming but optional for
voxcpm2-tts-streaming.

**Preset voices - no cloning, no handshake list.** The engine serves the named
preset voices in the `VM_AUDIOCPP_TTS_VOICES` manifest, selected per request via
the `voice` field. The shipped manifest carries:

- `designed_female_s101` - **the default**: calm adult female, even-paced,
  slightly synthetic (a starship-computer register).
- `designed_male_s202` - the male counterpart, deep and measured.
- `lcars_demo` - the LCARS computer voice, made from clips of Star Trek: The
  Next Generation. The audio is copyrighted by its owners and is included for
  personal, non-commercial use only (see the README).

An unknown `voice` name fails before anything is dispatched. The handshake's
`available_voices` (and `/info`) lists the preset names from the server's
manifest, so a consumer can discover them over the wire, and `default_voice`
names the preset a request without `voice` gets (see
[Voice selection](#voice-selection)). Voice cloning is not supported:
`POST /voice/upload` is rejected with `400` (see
[Voice cloning](#voice-cloning)).

**Per-model output rate.** Every reply's `audio_start` declares the rate the
sidecar's READY ack stated for the selected model_task: `48000` under
dots-tts-streaming and voxcpm2-tts-streaming (double VibeVoice's 24000 and the
highest rate the service emits), `24000` under breeze-tts-streaming and
qwen3-tts-streaming. There is no
single audiocpp rate. This is exactly why the
[per-reply sample-rate guarantee](#guarantees-you-can-depend-on) exists: open
your sink at the declared rate, never a hard-coded one, or replies play at the
wrong speed.

**Streaming delivery and `tts_stream` batching.** Synthesis is genuinely
streaming: PCM frames reach the client while the sidecar is still generating
(measured warm first-audio ~0.6 s for one-shot `tts`, and ~0.6 s from the chunk
that completes the first sentence on `tts_stream`). On `tts_stream` the engine
batches differently from the other engines, because each sidecar request
restarts prosody but text *within* one request flows through the model's
internal chunker with continuous vocoder state (the smooth-seam path):

- The **first completed sentence dispatches immediately** - dispatch is never
  gated on more text arriving.
- Everything that accumulates while a request is synthesizing coalesces into
  **one next request**, so a seam cannot occur at every sentence.
- After that first dispatch, the engine also waits up to
  `VM_AUDIOCPP_TTS_COALESCE_MS` (default 350 ms) for more sentences before
  synthesizing, then sends them together. A `done` flag ends the wait early, so
  the tail of a reply never stalls, and `tts_cancel` is honored during it.

**Why the wait exists.** Synthesis outruns arrival (measured RTF ~0.6), which
cuts the opposite way from what you might expect: a producer that forwards tokens
as they arrive completes one sentence at a time, and the engine is always idle and
waiting when it does, so *without* the wait every sentence becomes its own
request. That is the worst case, not a rare one - it is what any consumer
streaming an LLM token by token produces. It matters most under
breeze-tts-streaming, which re-derives the speaker from the reference audio on
every request, so a per-sentence split makes the voice shift audibly between
sentences rather than merely adding a seam. Measured across the same four
sentences, mean MFCC spread was 29.5 in one request against 48.6 split across
four, where 29.5 is the floor from differing phonetic content alone.

The wait is spent against the previous batch's playback rather than against
silence - at RTF ~0.6 a dispatched batch buys roughly 40% of its own duration in
slack before the client runs dry - so a few hundred ms does not gap the audio.
Set it to `0` to restore per-instant dispatch; raise it to trade first-audio
latency on later sentences for fewer boundaries.

The [forward-every-token rule](#gotchas) still applies verbatim, as do `flush`
and `done` semantics and the `VM_TTS_STREAM_FLUSH_TIMEOUT` idle flush. Long
sentences are cut at clause boundaries (commas), never mid-phrase, when they
exceed the model's 200-codepoint chunk budget.

Each internal chunk generates its own small utterance pause at the seam
(measured 400-900 ms untrimmed). The sidecar caps the silence gap at a seam to
`VM_AUDIOCPP_TTS_SEAM_SILENCE_MS` (default 400 ms), so seams pace like natural
sentence pauses; pauses the model renders *inside* one chunk are never touched,
and no voiced audio is ever dropped.

**Cancel semantics and measured latency.** The
[cancel contract](#turn-taking-protocol-surface) is unchanged - fire-and-forget
`tts_cancel`, exactly one `audio_end` with `reason: "cancelled"`, discard PCM
until it lands - but this engine honors it faster than the Python engines:
the `audio_end` ack and the halt of delivered audio are effectively immediate
(**under 1 ms** measured), because the sidecar is told to stop generating rather
than being left to run doomed. The sidecar stops within about one vocoder merge
group (~0.35 s of *generated, undelivered* audio), the warm session is reused,
and the next request's first audio lands in about **0.65 s** on the same
session - measured cancel-to-next-first-audio 622-669 ms across runs. The
"ack can lag by roughly a second" caveat in [Gotchas](#gotchas) does not apply
here; the stop-playback-immediately rule still does.

**Per-session VRAM cost.** Each WebSocket session gets its **own TTS sidecar
process** holding its own copy of the model: measured **~4.0 GiB of VRAM per
session** (dots q8_0 on an RTX 4060 Ti), released in full when the session
disconnects. A session running audio.cpp for both directions (STT Voxtral-4B +
TTS) holds ~7.6 GiB, so a 16 GiB card fits **one** dual-engine session - a
second does not fit. Like the audio.cpp STT engine, this is a far lower
concurrency ceiling than the in-process engines; if you need many simultaneous
TTS sessions, this engine is not the right choice today.

Session warm-up mirrors the STT engine: the sidecar is spawned and the model
loaded in the background when the WebSocket connects, so the first request on a
fresh connection may still pay part of that warm-up (~1.5 s to first audio
measured when the request races the load). Hold one long-lived connection
rather than reconnecting per utterance.

If the session's TTS sidecar process dies mid-request, the session receives an
`error` frame with `error_type: "tts_backend_fault"` and a `1011` close - the
TTS analog of `stt_backend_fault` in [Close codes](#close-codes); reconnecting
gets a fresh sidecar.

## Voice selection

For engines with preset voices (VibeVoice and audiocpp), the voice of a `tts` or
`tts_stream` request is chosen in this order:

1. **The request's `voice` field**, when it is set. An unknown name behaves as
   before: under audiocpp it fails the request; under VibeVoice the server logs
   a warning and speaks with the configured default (`VM_VIBEVOICE_VOICE`)
   instead. Pick names from `available_voices` to avoid both.
2. **The token's voice**, when the token that opened the connection has one and
   the running engine offers it. Set it with
   `voicemanager user token voice USER NAME VOICE` (clear it with `--clear`),
   with `voicemanager user token add USER NAME --voice VOICE`, or on the web
   tokens page. Both refuse a name the engine does not list.
3. **The engine's configured default**: `VM_VIBEVOICE_VOICE` or
   `VM_AUDIOCPP_TTS_VOICE`.

The handshake reports the result of steps 2 and 3 as `default_voice`: the voice
a request without `voice` gets on this connection. If the token's voice is not
offered by the running engine (the engine was switched or the preset removed),
the server skips it, uses the configured default, reports that default as
`default_voice`, and logs one warning per connection. A client therefore never
needs to know whether a token has a voice: read `default_voice`, and send
`voice` only to override it.

Under OpenVoice, which clones instead of using presets, `available_voices` is
empty, `default_voice` is `null` and a token's voice is ignored.

```text
token voice "narrator", audiocpp presets ["designed_female_s101", "narrator"]
server:  { "type": "ready", ..., "available_voices": ["designed_female_s101", "narrator"], "default_voice": "narrator" }
client:  { "type": "tts", "text": "Hello.", "request_id": "r1" }                  -> spoken as narrator
client:  { "type": "tts", "text": "Hello.", "request_id": "r2", "voice": "designed_female_s101" }  -> spoken as designed_female_s101
```

## The endpoint-hold consumption model

Streaming STT results arrive as `TranscriptionMessage` frames. On the
speculative endpoint-hold path (the nemotron STT engine, enabled when
`VM_STT_ENDPOINT_HOLD_MS` is greater than 0) a single spoken utterance is
revised in place across a short sequence of frames before it commits. This
section is the mental model a consuming client needs to render those revisions
correctly. Two `TranscriptionMessage` fields drive it: `endpoint` and
`is_final`.

### The three frame kinds for one utterance

Every frame for a given utterance carries the same `segment_index`. As the
speaker talks and then pauses, the server emits, in order:

1. **Interim partial** - `is_final=false`, `endpoint=false`. The live, still-
   growing hypothesis for the current segment. Expect many of these.
2. **Provisional endpoint frame** - `is_final=false`, `endpoint=true`. Emitted
   the moment the acoustic endpointer thinks the speaker stopped. It is a
   low-latency *guess* that the utterance is over: act on it early if you want,
   but it is **not** committed and may still be revised if speech resumes within
   the hold window.
3. **Confirmed final** - `is_final=true`, `endpoint=false`. The immutable commit,
   emitted once a hold window's worth of post-endpoint audio has passed with no
   resumed speech. This is the last word for that `segment_index`; note
   `endpoint` is back to `false` here - the endpoint hint has served its purpose.

If speech resumes during the hold, the server cancels the pending commit, folds
the held text back into the growing hypothesis, and keeps emitting interim
partials under the **same** `segment_index` - so the provisional guess is
transparently walked back and re-confirmed later.

### The upsert-by-index revision contract

Because all frames for one utterance share a `segment_index`, a client must
treat that index as a **stable key into a revisable slot**, not as an append
cursor. The contract is upsert-by-index: keep a map from `segment_index` to the
latest text/state, and for each incoming frame **replace** the entry at its
index with the newer frame. A provisional endpoint frame overwrites the interim
text at that index; the confirmed final overwrites it once more. Rendering is
then simply "show the latest value at each index," and the speculative endpoint
correction happens for free.

For example, one utterance at `segment_index: 3`:

```text
{ "segment_index": 3, "text": "book me a flight to",       "is_final": false, "endpoint": false }
{ "segment_index": 3, "text": "book me a flight to boston", "is_final": false, "endpoint": false }
{ "segment_index": 3, "text": "book me a flight to boston", "is_final": false, "endpoint": true  }  <- provisional
{ "segment_index": 3, "text": "book me a flight to boston", "is_final": true,  "endpoint": false }  <- committed
```

The consumer holds one slot for index 3 and overwrites it four times; the final
rendered value is the committed text.

### endpoint vs is_final

These two booleans answer different questions and must not be conflated:

- **`endpoint`** is a *latency hint*: "the speaker probably just stopped - you
  may act now." It is speculative and reversible. On the endpoint-hold path it is
  `true` only on the provisional frame; on the whisper and qwen engines it is
  always `false`.
- **`is_final`** is a *commitment*: "this text for this `segment_index` will not
  change." Only act irreversibly (e.g. dispatch the utterance to a downstream
  agent) on `is_final=true`, or knowingly accept the speculative risk of acting
  on `endpoint=true` for lower latency.

A frame can therefore be an endpoint but not yet final (the provisional guess),
or final without ever having been flagged an endpoint (the whisper/qwen path,
and the endpoint-hold path's own confirmed final where `endpoint` is `false`
again).

## Turn-taking protocol surface

The `realtime-turn-taking` work landed six behaviours on top of the base
STT/TTS protocol, all **additive and opt-in** - a client that ignores them keeps
the pre-turn-taking behaviour unchanged. Each behaviour below names the source
symbol that defines it, so you can diff the contract against current code.

### 1. Flush a streaming reply at a client-detected boundary

A `tts_stream` frame carries a `flush` flag (`TTSStreamMessage.flush`,
default `false`). Setting `flush: true` tells the server to synthesize
everything it has accumulated for that `request_id` **right now** - complete
sentence or not - and leave the request **open** for more text. It is distinct
from `done`: `done` closes the request and yields a trailing `audio_end`, while
`flush` never queues the end-of-sentence sentinel and never yields an
`audio_end`. The Python client exposes this as `flush_tts(request_id)`, which
puts exactly one `flush: true, done: false` frame on the wire.

```json
{ "type": "tts_stream", "request_id": "b7c1d0e2-9f3a-4c58-8e21-0a1b2c3d4e5f", "text": "Booking your flight now.", "flush": true, "done": false }
```

Flush only at a boundary **you** have determined is the end of a speakable
unit; flushing mid-word produces a clipped, unnatural segment seam.

### 2. Why an audio stream ended: `reason` on `audio_end`

`AudioEndMessage` carries a `reason` field (`AudioEndMessage.reason`, default
`"complete"`). It tells a client *why* a TTS reply stopped:

- `"complete"` - the reply finished normally. This is every pre-existing close
  path, including the `VM_TTS_STREAM_FLUSH_TIMEOUT` idle-flush close, which is a
  normal completion, not an interruption.
- `"cancelled"` - a `tts_cancel`, or a supersede by a newer `request_id`,
  stopped the reply (see behaviour 5).
- `"error"` - synthesis failed after the stream opened: the request is
  over and delivered no further audio. The session survives a request-scoped
  failure - the next `tts` on the same connection proceeds normally. (A
  backend-fatal fault still closes the whole session with an `error` frame and
  close code 1011, as before.)
- `"interrupted"` - reserved for future server-side barge-in; never emitted
  today.

A client that does not read `reason` sees an `audio_end` and treats the reply as
over, exactly as before.

```json
{ "type": "audio_end", "request_id": "b7c1d0e2-9f3a-4c58-8e21-0a1b2c3d4e5f", "reason": "cancelled" }
```

### 3. `turn_id` on transcription frames

Every `TranscriptionMessage` carries a `turn_id` (`TranscriptionMessage.turn_id`,
default `0`). It identifies the turn an utterance **belongs to** - the turn that
was open when the utterance's *audio was received*, not the turn current when
the frame is sent. That distinction matters when a final commits across a turn
boundary: the frame keeps the `turn_id` of the audio it transcribes. A
connection that never opens a turn leaves every frame at `turn_id: 0`. The
Python client guarantees the key is present on every transcription dict via
`_stamp_turn_id`, so consumers can read `frame["turn_id"]` unconditionally.

```json
{ "type": "transcription", "text": "book me a flight to boston", "is_final": true, "turn_id": 4, "segment_index": 3 }
```

### 4. The `turn_start` / `turn_started` control frames

A client opens a new conversation turn by sending the bare control frame
`{"type": "turn_start"}` (no payload). The **server owns the turn counter**: it
increments the count (`turn_tracker.advance()` in the WS handler) and replies
with a `turn_started` ack (`TurnStartedMessage`) carrying the newly opened
`turn_id`. The client learns which `turn_id` to filter on from the ack rather
than inferring it from the next stamped transcription frame. Both message types
are declared on `MessageType` (`turn_start`, `turn_started`). The Python client
wraps the round-trip as `turn_start()`, which sends the frame and returns the
server-assigned id.

```text
client -> { "type": "turn_start" }
server -> { "type": "turn_started", "turn_id": 5 }
```

Opening turns is optional; it exists so a consumer can fence out
now-stale utterances from a previous turn after a barge-in.

### 5. Cancel: an immediate `audio_end`, then no PCM

Cancelling a TTS reply is fire-and-forget by design. The client sends a
`tts_cancel` frame (`TTSCancelMessage`) - or the Python client's
`cancel_tts(request_id)` - and the server acks **immediately** with a single
`audio_end` carrying `reason: "cancelled"`. It does not wait for the synthesis
to stop, because the synthesis is blocked in an executor thread that cannot be
interrupted; that doomed task keeps running and will later try to emit its own
audio and its own `audio_end`, all of which the server silently drops. The
contract a client can therefore rely on is: **exactly one `audio_end` per
`request_id`, and no PCM after it.**

Because the ack can arrive while binary frames are still in flight, a client must
discard PCM for a cancelled `request_id` until its `audio_end` lands; the
[Gotchas](#gotchas) section covers this rule and the client's `_cancelled_tts`
mechanism in full.

### 6. A cancelled `request_id` is permanently dead

Once a `request_id` is cancelled, it is **dead for the life of the connection**.
A later `tts` or `tts_stream` frame carrying that same id is ignored outright by
the server's `TTSSession`, so a token that was already on the wire when the
client cancelled cannot reopen the request the client just killed. The server
keeps a bounded ring of recently cancelled ids for this purpose (the most recent
window is the only place a stale token can still be in flight). The practical
rule for clients: **never reuse a `request_id` after cancelling it** - always
mint a fresh id (e.g. a new UUID) for the next reply.

## WebSocket message-schema reference

All WebSocket **text** frames are JSON objects with a `type` field that routes
them; every valid value is a member of `MessageType`. Each frame below maps to a
Pydantic model in `protocol.py` (named in its heading), except the bare
`turn_start` control frame, which has no model. Field types are given as JSON
types; the **Default** column is the value a field takes when the sender omits
it, and *(required)* marks a field with no default. Binary frames carry no JSON
envelope - see [Binary frames](#binary-frames) at the end of this section.

`MessageType` also declares `audio_chunk`, which is **reserved and currently
unused**: no frame of that type is ever sent in either direction. STT input and
TTS output audio both travel as binary PCM, not as an `audio_chunk` JSON frame.

### Server to client frames

#### `ready` - `HandshakeMessage`

The first frame the server sends after accepting the connection. Announces the
session id and the engines in use.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | `"ready"` | Frame discriminator. |
| `session_id` | string | *(required)* | Server-assigned session id; pass it to `POST /voice/upload`. |
| `audio_format` | object | *(required)* | Expected input PCM format, e.g. `{"sample_rate": 16000, "channels": 1, "bit_depth": 16, "encoding": "pcm"}`. |
| `stt_engine` | string | `"whisper"` | Active STT engine identifier (e.g. `"whisper"`, `"qwen"`); informational only. |
| `tts_engine` | string | `"openvoice"` | Active TTS engine (`"openvoice"`, `"vibevoice"`, or `"audiocpp"`). |
| `available_voices` | array of string | `[]` | Preset voice names, sorted, for VibeVoice and audiocpp (whose presets are described in [its section](#the-audiocpp-tts-engine-presets-per-model-sample-rate-one-request-batching)); empty for OpenVoice. |
| `default_voice` | string or null | `null` | The voice a `tts` or `tts_stream` request without `voice` gets on this connection: the token's voice when the engine offers it, else the engine's configured default. `null` under OpenVoice. See [Voice selection](#voice-selection). |

#### `transcription` - `TranscriptionMessage`

One STT result. Interim and final results share the schema; see
[The endpoint-hold consumption model](#the-endpoint-hold-consumption-model) for
how `is_final`, `endpoint`, and `segment_index` interact.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | `"transcription"` | Frame discriminator. |
| `text` | string | *(required)* | The transcribed text for this segment. |
| `is_final` | boolean | `true` | `true` once the text for this `segment_index` is committed and will not change. |
| `endpoint` | boolean | `false` | Speculative end-of-utterance hint; `true` only on the endpoint-hold path's provisional frame. |
| `turn_id` | integer | `0` | The turn this utterance belongs to (the turn open when its audio was received). |
| `segment_index` | integer | `0` | Stable key for the utterance across its revisions (upsert by this, do not append). |
| `language` | string | `""` | Detected language code, when the engine reports one. |
| `inference_duration` | number | `0.0` | Seconds the engine spent producing this result. |
| `confidence` | number | `0.0` | Engine confidence, when reported. |
| `start` | number | `0.0` | Utterance start time in seconds. |
| `end` | number | `0.0` | Utterance end time in seconds. |

#### `audio_start` - `AudioStartMessage`

Sent immediately before the first PCM chunk of a TTS reply. Carries the output
audio format so the client can open a sink at the right sample rate.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | `"audio_start"` | Frame discriminator. |
| `request_id` | string | *(required)* | The `request_id` this audio belongs to; match it against your request. |
| `sample_rate` | integer | *(required)* | Output rate in Hz (e.g. `22050` OpenVoice, `24000` VibeVoice; audiocpp per model_task: `48000` dots/voxcpm2, `24000` breeze/qwen3). |
| `channels` | integer | `1` | Channel count. |
| `bit_depth` | integer | `16` | Bits per sample. |
| `encoding` | string | `"pcm"` | Sample encoding. |

#### `audio_end` - `AudioEndMessage`

Marks the end of one TTS reply. Exactly one `audio_end` is emitted per
`request_id`; no PCM follows it.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | `"audio_end"` | Frame discriminator. |
| `request_id` | string or null | `null` | The completed reply's `request_id`. |
| `reason` | string | `"complete"` | Why the reply ended: `"complete"`, `"cancelled"`, or (reserved) `"interrupted"`. See [Turn-taking](#turn-taking-protocol-surface). |

#### `turn_started` - `TurnStartedMessage`

The server's ack to a `turn_start` control frame, reporting the newly opened
turn's id. The server owns the turn counter.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | `"turn_started"` | Frame discriminator. |
| `turn_id` | integer | *(required)* | The id of the turn just opened; filter subsequent utterances by it. |

#### `error` - `ErrorMessage`

Sent before the server closes on a protocol violation, and for non-fatal notices
such as the inactivity `timeout_warning`. See [Close codes](#close-codes).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | `"error"` | Frame discriminator. |
| `error_type` | string | *(required)* | Machine-readable category (e.g. `"timeout_warning"`). |
| `message` | string | *(required)* | Human-readable detail. |

### Client to server frames

#### `tts` - `TTSRequestMessage`

Request one-shot synthesis of a complete string.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | `"tts"` | Frame discriminator. |
| `text` | string | *(required)* | Text to synthesize. |
| `request_id` | string | *(required)* | Client-minted id (e.g. a UUID) correlating the reply's `audio_start`/`audio_end`. |
| `voice` | string or null | `null` | Preset voice name (VibeVoice and audiocpp); ignored under OpenVoice. When `null`, the connection's `default_voice` is used (see [Voice selection](#voice-selection)). |

#### `tts_stream` - `TTSStreamMessage`

Append text to a streaming reply accumulated by `request_id`, or control it via
`done`/`flush`. See [Turn-taking](#turn-taking-protocol-surface) for `flush`
semantics.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | `"tts_stream"` | Frame discriminator. |
| `request_id` | string | *(required)* | The streaming reply's id; reuse it across chunks. |
| `text` | string | *(required)* | The next text fragment to append (`""` on a pure control frame). |
| `done` | boolean | `false` | `true` closes the request; the server synthesizes the remainder and emits a trailing `audio_end`. |
| `flush` | boolean | `false` | `true` synthesizes the buffer so far and leaves the request open; no `audio_end`. |
| `voice` | string or null | `null` | Preset voice (VibeVoice and audiocpp); honored on the first frame, which fixes the reply's voice. |

#### `tts_cancel` - `TTSCancelMessage`

Cancel an in-flight TTS reply. The server acks with an immediate
`audio_end` carrying `reason: "cancelled"`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | `"tts_cancel"` | Frame discriminator. |
| `request_id` | string | *(required)* | The reply to cancel. Never reuse this id afterward (it is permanently dead). |

#### `turn_start` - bare control frame (no model)

Open a new conversation turn. Send the frame verbatim, with no other fields; the
server replies with a `turn_started` ack.

```json
{ "type": "turn_start" }
```

### Binary frames

Audio never travels as a JSON frame. In both directions it is sent as raw binary
WebSocket messages carrying little-endian 16-bit PCM samples:

- **Client to server** - STT input: 16 kHz, mono, 16-bit PCM (the `audio_format`
  the handshake advertised). The first binary frame is format-validated; a
  mismatch closes the socket (see [Close codes](#close-codes)).
- **Server to client** - TTS output: the format announced by the preceding
  `audio_start` (rate is engine-dependent), framed between `audio_start` and
  `audio_end`.

## Client method reference

`VoiceManagerClient` (`client.py`) is the reference Python client and one
concrete implementation of the protocol above. It holds a single persistent
WebSocket per `async with` scope; all STT and TTS calls share that connection.
Every method is a coroutine (`await` it) unless marked. `stream_stt_live` and
`stream_tts_live` are async generators (`async for`); `save_tts_to_file` and
`pcm_to_wav_bytes` are `@staticmethod` and take no connection.

### Construction and lifecycle

| Symbol | Signature | Purpose |
|---|---|---|
| `VoiceManagerClient` | `(host="localhost", port=8000, token="", max_retries=3, use_ssl=False)` | Construct a client. No socket opens until you enter the context manager. |
| `async with client:` | `__aenter__` / `__aexit__` | Opens the WebSocket, reads the `ready` handshake, starts the single reader task; closes the socket and drains queues on exit. |

### Speech to text

| Method | Signature | Returns | Purpose |
|---|---|---|---|
| `stream_stt` | `(audio_chunks)` | `list[dict]` | Send every PCM chunk, then collect and return all transcription dicts the server emits. |
| `stream_stt_live` | `(audio_chunks)` | `AsyncIterator[dict]` | Send PCM and yield each transcription dict (interim and final) as it arrives. |
| `transcribe_file` | `(path, chunk_frames=4000)` | `list[dict]` | Convenience: read a 16 kHz mono WAV as raw PCM frames and stream it via `stream_stt`. |

### Turn taking

| Method | Signature | Returns | Purpose |
|---|---|---|---|
| `turn_start` | `(timeout=5.0)` | `int` | Send the `turn_start` control frame, await the `turn_started` ack, and return the server-assigned `turn_id`. |

### Text to speech

| Method | Signature | Returns | Purpose |
|---|---|---|---|
| `synthesize_tts` | `(text, voice=None)` | `tuple[bytes, int]` | One-shot synthesis; returns `(pcm_bytes, sample_rate)`. |
| `stream_tts` | `(text_chunks, voice=None)` | `tuple[bytes, int]` | Stream text chunks, collect all audio, and return `(pcm_bytes, sample_rate)`. |
| `stream_tts_live` | `(text_chunks, voice=None, request_id=None)` | `AsyncIterator[bytes]` | Stream text and yield PCM chunks as they arrive; supply or read `request_id` for barge-in. |
| `cancel_tts` | `(request_id)` | `None` | Fire-and-forget cancel; returns as soon as the `tts_cancel` frame is sent. |
| `flush_tts` | `(request_id)` | `None` | Synthesize the buffered text for an in-flight streaming request now, leaving it open (one `flush:true, done:false` frame). |

### Voice cloning and REST

| Method | Signature | Returns | Purpose |
|---|---|---|---|
| `upload_voice` | `(audio_data, filename="voice.wav")` | `dict` | Upload a WAV sample for OpenVoice cloning on this session (`POST /voice/upload`). |
| `health` | `()` | `dict` | `GET /health` - model readiness. |
| `info` | `()` | `dict` | `GET /info` - loaded models, device, and active sessions. |

### Static audio helpers

| Method | Signature | Returns | Purpose |
|---|---|---|---|
| `save_tts_to_file` | `(pcm_data, path, sample_rate, channels=1)` | `None` | Write raw PCM bytes to a WAV file. |
| `pcm_to_wav_bytes` | `(pcm_data, sample_rate, channels=1)` | `bytes` | Wrap raw PCM bytes in a WAV container in memory. |

### Instance attributes worth reading

These public attributes are set by the client and are part of its contract; the
turn-taking and barge-in flows above rely on them.

| Attribute | Type | Meaning |
|---|---|---|
| `session_id` | `str \| None` | Server-assigned session id from the handshake; needed for `upload_voice`. |
| `stt_engine` / `tts_engine` | `str` | Engine identifiers reported in the handshake. |
| `available_voices` | `list[str]` | Preset voices offered by the server (VibeVoice and audiocpp). |
| `default_voice` | `str \| None` | The voice a TTS request without `voice` gets on this connection; `None` under OpenVoice. |
| `connected` | `bool` | Whether the reader task holds a live connection. |
| `reconnected` | `bool` | Set `true` after a transparent reconnect; the voice profile must be re-uploaded, after which it clears. |
| `last_tts_sample_rate` | `int` | Output rate of the most recent TTS reply, updated from `audio_start` during live streaming. |
| `last_tts_request_id` | `str \| None` | The `request_id` of the most recently started TTS request; pass it to `cancel_tts` for barge-in. |

## Close codes

The server closes the WebSocket with a specific code per failure class. Except
for `1008`, every close is preceded by an `error` frame (`ErrorMessage`) whose
`error_type` names the machine-readable cause, so a client can branch on that
before the socket drops.

| Code | `error_type` | Cause | What a client should do |
|---|---|---|---|
| `1008` | *(none)* | No valid bearer token. The server closes with `1008` **before** accepting the upgrade, so the client never receives a `1008` frame - it sees an **HTTP 403** handshake rejection instead. | Fix the token; do not retry with the same one. `VoiceManagerClient` surfaces this as `AuthenticationError`, not a close event. |
| `1003` | `invalid_audio_format` | The first binary frame was not 16 kHz mono 16-bit PCM (nor a WAV container). | Send correctly formatted PCM (see [Binary frames](#binary-frames)); reconnect. |
| `1001` | `timeout` | Inactivity: no traffic for `VM_SESSION_TIMEOUT` seconds. A `timeout_warning` `error` frame arrives ~30s ahead of the close. | Keep the session active or reconnect; heed the `timeout_warning`. |
| `1011` | `server_error` | An unexpected server-side exception in the connection handler. | Reconnect; if it persists, inspect the server logs. |
| `1011` | `stt_backend_fault` | audio.cpp engine only. The STT sidecar process faulted mid-session (a runtime fault in the native model, or the process died). The backend was serving and broke; the VoiceManager server itself is unaffected, as are other sessions. | Reconnect - a fresh session gets a fresh sidecar, so a retry usually succeeds. |
| `1011` | `tts_backend_fault` | audio.cpp TTS engine only. This session's TTS sidecar process died mid-request. Per-session sidecar means per-session blast radius: only this session closes. | Reconnect - a fresh session gets a fresh sidecar, so a retry usually succeeds. |
| `4001` | `stt_backend_unavailable` | audio.cpp engine only. The STT sidecar could not be brought up for this session at all, and only an operator can fix it: a missing or unexecutable sidecar binary (`spawn_failed`), a missing or unreadable model file (`load_failed`), an unknown model/task id (`unknown_model_task`), a protocol mismatch (`unsupported_protocol`), no sidecar reachable at the configured socket (`dial_failed`), or nothing configured to dial or spawn (`no_sidecar`). The `message` carries the specific cause code. | Do NOT retry in a loop - this is a server-side configuration problem and every reconnect will fail the same way. Surface it and have an operator check the `VM_AUDIOCPP_*` settings. |
| `4002` | `stt_backend_unready` | audio.cpp engine only. The sidecar could not take *this* session *right now*, but nothing is misconfigured - either `at_capacity` (the server already has `VM_AUDIOCPP_MAX_SIDECARS` concurrent sessions; see the per-session VRAM cost above) or `busy` (the sidecar already has a live session - one session per sidecar process). Any sidecar error code not on the `4001` list above also lands here. The `message` carries the specific cause code. | Retry with backoff; the condition typically clears when another session ends. Do not page an operator on this one. |

The three audio.cpp codes are deliberately different because they ask different
things of a client: `1011` means the backend was serving and broke, so a fresh
session may well work; `4002` means "not right now, try again shortly"; `4001`
means this server cannot serve the engine at all and retrying is futile. Only
`4001` carries the do-not-retry contract, and it is an explicit list of
configuration faults - an unrecognised sidecar error code is reported as `4002`,
so an unclassified fault can never strand a client that would otherwise have
recovered. `4001`/`4002` sit in the RFC 6455 private `4000-4999` range, so they
can never be confused with a transport-level close.

**A recoverable mid-session backend error does not close the session at all.**
If the sidecar reports a recoverable failure while decoding one utterance
(`internal`, which it emits from a caught exception and then keeps serving), the
server drops **that utterance's final** and the session streams on - no `error`
frame, no close. The client-visible effect is one utterance that produces no
`transcription`; the cause is logged server-side. Ending a healthy session over
a single bad utterance would cost far more than the utterance does.

Why `1008` is a special case is expanded under [Gotchas](#gotchas): auth is
verified before `accept()`, so a bad token can only manifest as a 403, never as a
`1008` close.

## Configuration

The client integrator connects to a running server; these are the `VM_`
environment variables (read by `config.py`, prefix `VM_`) that shape the
client-visible contract - which host to reach, which engines are active, and the
timeouts that end a session. Defaults are the server's built-in values.

| Env var | Default | Why it matters to a client |
|---|---|---|
| `VM_DB_PATH` | `$XDG_STATE_HOME/voicemanager/state.db` (else `~/.local/state/voicemanager/state.db`) | The SQLite file that holds users and tokens. The CLI and the server read the same setting, so a token made with `voicemanager user token add` works on the server that uses the same path. |
| `VM_HOST` | `0.0.0.0` | Bind address; the host the client points at. |
| `VM_PORT` | `8000` | Port the client connects to (`/ws/stream`, REST). |
| `VM_STT_ENGINE` | `whisper` | Active STT engine (`whisper`, `qwen`, `nemotron`, or `audiocpp`); only `nemotron` drives the [endpoint-hold path](#the-endpoint-hold-consumption-model). |
| `VM_AUDIOCPP_MODEL_TASK` | `nemotron-3.5-asr-streaming` | Which model the `audiocpp` engine serves; selects the transcription cadence a client sees (both registered ASR model_tasks, `nemotron-3.5-asr-streaming` and `voxtral-mini-4b-realtime-asr-streaming`, stream growing partials mid-speech). `VM_AUDIOCPP_MODEL` must point at the matching GGUF. |
| `VM_TTS_ENGINE` | `openvoice` | Active TTS engine (`openvoice`, `vibevoice`, or `audiocpp`); voice cloning is OpenVoice-only, handshake `available_voices` lists the VibeVoice or audiocpp presets, audiocpp serves [named presets](#the-audiocpp-tts-engine-presets-per-model-sample-rate-one-request-batching) at its model_task's rate (48 kHz dots/voxcpm2, 24 kHz breeze). |
| `VM_VIBEVOICE_VOICE` | `Carter` | Default preset voice when neither the request nor the token names a voice (VibeVoice). |
| `VM_AUDIOCPP_TTS_VOICE` | `designed_female_s101` | Default preset voice when neither the request nor the token names a voice (audiocpp); the full `VM_AUDIOCPP_TTS_*` set is documented [in its section](#the-audiocpp-tts-engine-presets-per-model-sample-rate-one-request-batching). |
| `VM_STT_ENDPOINT_HOLD_MS` | `1000` | Speculative endpoint-hold window; `> 0` enables provisional `endpoint` frames, `0` disables them (immediate finals). |
| `VM_SESSION_TIMEOUT` | `300` | Inactivity seconds before a `1001` close (with a ~30s `timeout_warning`). |
| `VM_TTS_STREAM_FLUSH_TIMEOUT` | `10` | Idle seconds before a streaming reply auto-flushes and completes with `reason: "complete"`. |
| `VM_DEVICE` | `auto` | Compute device (`auto`, `cpu`, `cuda`, `cuda:N`); affects latency, not the protocol. |

The server has many more `VM_` variables for model selection and tuning
(Whisper, Qwen, Nemotron, OpenVoice, and VibeVoice parameters). Those are
server-operator concerns that do not change the wire contract; the full,
authoritative list lives in `config.py` and `.env.example`.

## Voice cloning

Voice cloning is an **OpenVoice-only** capability (`VM_TTS_ENGINE=openvoice`). A
client uploads a short WAV sample; the server extracts an OpenVoice *tone-color*
embedding (`target_se`) from it and stores that embedding on the uploading
**session's** `TTSPipeline`. Every subsequent `tts` / `tts_stream` reply on that
same session is then rendered in the cloned voice. Under the VibeVoice and
audiocpp engines the upload is rejected with `400` (see below); both offer
preset voices via the `voice` field instead - neither clones. The audiocpp
rejection message names the alternative: select a preset from its manifest, or
run `VM_TTS_ENGINE=openvoice` for upload cloning.

The flow:

1. **Open a WebSocket** and read the `session_id` from the `ready` handshake -
   the clone is bound to that session.
2. **POST the sample** to `/voice/upload` with the `Authorization: Bearer`
   header, `session_id` as a query parameter and the WAV as a multipart `file`
   field. The session must belong to the token's user. Send at least a few
   seconds of clear speech; too short a clip is rejected `400`. The Python client
   wraps this as `upload_voice(audio_data, filename="voice.wav")`.
3. **The server extracts and stores** the tone color on the session's pipeline
   (overwriting any previous profile) and returns `{"status": "voice profile
   updated"}`.
4. **Synthesize as usual** on that session; replies now use the cloned voice.

Two properties a client must plan for:

- **Per-session and volatile.** The profile lives only on that session's
  in-memory pipeline. It is lost on disconnect and never shared across sessions.
- **Re-upload after reconnect.** A transparent reconnect starts a fresh session
  with no profile; `VoiceManagerClient` sets `reconnected = True` to flag this,
  and clears it once `upload_voice` re-establishes the profile.

```text
client:  ws connect        -> ready { "session_id": "3f2b1a90-5c7e-4d21-9b0a-1e2f3a4b5c6d", ... }
client:  POST /voice/upload?session_id=3f2b1a90-5c7e-4d21-9b0a-1e2f3a4b5c6d  (Authorization: Bearer vmp_..., multipart file=voice.wav)
server:  200 { "status": "voice profile updated" }
client:  { "type": "tts", "text": "This now speaks in the cloned voice.", "request_id": "..." }
```

## REST endpoints

The service exposes three REST endpoints alongside the WebSocket. `/health` is
open; `/info` and `/voice/upload` need the `Authorization: Bearer` header and
answer `401` without a valid token (see [Authentication](#authentication)).

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | none | Model readiness. |
| `GET` | `/info` | Bearer token | Loaded models, device, and your active sessions. |
| `POST` | `/voice/upload` | Bearer token | Upload a voice sample for OpenVoice cloning. |

### `GET /health`

Reports which models are loaded. `null` for any slot that is not loaded.

```json
{ "status": "ok", "models": { "stt": "faster-whisper", "vad": "silero-vad", "tts": "openvoice-v2" } }
```

The `stt` value reflects the active engine: `"faster-whisper"`, `"qwen3-asr"`,
`"nemotron-streaming"`, or - under the audiocpp engine - the active model_task id
(e.g. `"voxtral-mini-4b-realtime-asr-streaming"`), which names the model family
the sidecar is serving. `tts` is `"openvoice-v2"`, `"vibevoice-realtime (...)"`,
or - under the audiocpp engine - `"audiocpp (...)"` naming the loaded GGUF.

### `GET /info`

Fuller introspection: engines, model names, compute device, TTS sample rate,
preset voices, and a list of the caller's active session summaries.

```json
{
  "stt_engine": "whisper",
  "tts_engine": "openvoice",
  "models": { "stt": "tiny", "tts": "openvoice-v2" },
  "device": "cuda:3",
  "whisper_model": "tiny",
  "qwen_model": null,
  "nemotron_model": null,
  "audiocpp_model": null,
  "audiocpp_model_task": null,
  "tts_model": "openvoice-v2",
  "tts_sample_rate": 22050,
  "available_voices": [],
  "sessions": []
}
```

`tts_sample_rate` is the output rate of the active TTS engine; `available_voices`
lists the VibeVoice or audiocpp presets and is empty for OpenVoice; `sessions`
lists the active sessions opened with tokens of the calling user (empty when
none). Other users' sessions are never listed.

Under the audiocpp engine, `models.stt` is the basename of the loaded GGUF
(e.g. `"voxtral-mini-4b-realtime-2602-q4_k.gguf"` - the full path stays
server-side), `audiocpp_model` carries the same basename, and
`audiocpp_model_task` is the active model_task id (the selection key documented
above). All three are `null` when the audiocpp engine is not the active one.

Under the audiocpp **TTS** engine, `tts_model` is the basename of the TTS GGUF
(falling back to the model_task id when no model path is set),
`tts_sample_rate` is the configured model_task's output rate (`48000` for dots
and voxcpm2, `24000` for breeze-tts-streaming and qwen3-tts-streaming; the startup
engine never opens a
sidecar, so the value mirrors the sidecar registry, and `audio_start` on each
reply carries the rate the sidecar's READY actually stated), and
`available_voices` lists the preset names from the manifest (see
[its section](#the-audiocpp-tts-engine-presets-per-model-sample-rate-one-request-batching)).

### `POST /voice/upload`

Header: `Authorization: Bearer <token>`. Query parameter: `session_id` (the
session to bind the clone to; it must belong to the token's user). Body:
multipart form with a `file` field containing the WAV sample.

| Status | When |
|---|---|
| `200` | Success - body `{"status": "voice profile updated"}`. |
| `400` | The server runs VibeVoice or audiocpp (cloning unsupported - presets only), or the sample is empty or too short. |
| `401` | No `Authorization: Bearer` header, or the token is unknown or revoked. |
| `404` | `session_id` is not a live session of the token's user (an unknown id and another user's session answer the same). |
| `503` | The session registry or TTS engine is unavailable. |

## Gotchas

### Forward every token verbatim - do not batch into sentences

The counterintuitive rule for streaming TTS is: **forward every token to
`tts_stream` the instant it arrives, exactly as you received it.** Do not
accumulate tokens into a complete sentence before sending. The tidy-looking
"buffer until I have a full sentence, then send it" strategy makes latency
*worse*, not better.

The cause is server-side and unavoidable. The server's streaming TTS pipeline
(`TTSStreamPipeline`) splits everything it has buffered for the request into
sentences and dispatches all-but-the-last, holding the final one back - it cannot
know a trailing fragment is a finished sentence until more text or a close
arrives. So when a client hands it one already-complete sentence, the split
yields a single sentence, nothing is dispatched, and that lone sentence sits in
the server's buffer until the next token or `done`. By pre-assembling the
sentence the client has only moved the wait to its own side and added to it.

**When to flush.** If your client can detect its own speakable boundaries, that
is exactly what the `flush` flag is for (see
[Turn-taking, behaviour 1](#turn-taking-protocol-surface)). Send `flush: true` -
the Python client's `flush_tts(request_id)` - at a boundary you have determined
is the end of a speakable unit to make the server synthesize the held buffer
immediately while leaving the request open. Flush *only* at such a boundary;
flushing mid-word produces a clipped, unnatural segment seam. Flushing is the
sanctioned way to cut first-audio latency - client-side batching is not.

### After `tts_cancel`, discard PCM until `audio_end` - and stop playback now

Two rules govern the window after you send a `tts_cancel`, and a client needs
both.

**Discard binary frames until this request's `audio_end`.** PCM for the
cancelled reply that was already in flight over TCP keeps arriving after the
cancel and is **stale** - it belongs to the turn the human just interrupted, and
nothing in the binary frame itself marks it as such. Play it and the console
speaks the interrupted reply for a beat. So from the moment you issue the cancel
until that `request_id`'s `audio_end` (carrying `reason: "cancelled"`) lands,
drop every binary frame for it. The Python client does this with the
`_cancelled_tts` set: `cancel_tts(request_id)` adds the id, the receive path
drops binary frames for it, and the id is cleared when the `audio_end` arrives.

**Do not wait for `audio_end` to consider the turn over - stop local playback
immediately.** The ack can lag the cancel by roughly a second, because synthesis
is blocked in an executor thread that asyncio cannot preempt. A client that
treated `audio_end` as the stop signal would sit in a SPEAKING state for that
whole second after the user interrupted. Stop your own audio output the instant
you send the cancel; treat the eventual `audio_end` as **cleanup** (the cue to
clear the discard flag), not as the signal that the turn has ended. (Under the
audiocpp engine the ack is effectively immediate - measured under 1 ms - but
both rules stay correct there, so implement them unconditionally.)

### A bad token is an HTTP 403 at the handshake, not close code 1008

Auth is checked **before** the socket is accepted. Without a valid token the handler
calls `websocket.close(code=1008)` and returns *before* `accept()` - so no
WebSocket is ever established, and the ASGI server rejects the upgrade with a
plain **HTTP 403 Forbidden**. The `1008` close code never reaches the client.

The trap: a client that handles auth failure only on the close-code path - i.e.
that waits to connect successfully and then inspect a close frame for `1008` -
will **never** see the rejection. Its `connect()` raises an HTTP 403 at handshake
time instead, and a listener watching for a close code is never called. Handle
the 403 on the connect/handshake call itself; treat it, not `1008`, as the
"bad token" signal. The `1008` in [Close codes](#close-codes) is the code the
server *names*, but it is only observable for post-accept auth failures, which do
not currently exist - every bad token fails at the handshake.

### Track turns with `turn_id`, never a `segment_index` watermark

`segment_index` is monotonic **per connection**, not per turn - it never resets,
and on a long-lived socket it climbs for the life of the link. The tempting fix
for "only look at the current turn" is a high-water mark: record the highest
`segment_index` seen at some boundary, then ignore any frame whose index is not
above it. **Do not do this** - it is structurally broken against the endpoint-hold
contract.

The reason is the [upsert-by-index revision contract](#the-upsert-by-index-revision-contract):
one utterance's frames all share a single `segment_index`, and later frames
*revise* that same slot in place - the provisional endpoint frame and the
confirmed final both carry the dispatched segment's index. A watermark parked at
that index silently filters out the endpoint-hold's **own revision** of the
segment, because the revised frame reuses an index that is no longer above the
mark. A consumer that dispatches on the endpoint frame and wants to supersede
that run when the confirmed final arrives finds the final filtered out as "old" -
the watermark blocks the very supersede it was built to enable.

Use `turn_id` for turn scoping instead (see
[Turn-taking, behaviours 3 and 4](#turn-taking-protocol-surface)). Open a turn
with `turn_start`, take the `turn_id` from the `turn_started` ack, and filter
transcription frames by that id. This fences out stale prior-turn utterances
without ever discarding an in-place revision of the current turn's own
segments - which is exactly what the watermark gets wrong.

## Guarantees you can depend on

These are properties the protocol commits to keep, so a client may build on them.

**The output sample rate is declared per turn on `audio_start` - never hard-code
it.** Every TTS reply opens with an `audio_start` carrying `AudioStartMessage.sample_rate`,
and that rate depends on the engine and can differ from one reply to the next -
`22050` Hz under OpenVoice, `24000` Hz under VibeVoice, and under the audiocpp
engine whatever its model_task's READY declared (`48000` for dots and voxcpm2,
`24000` for breeze and qwen3). Open your audio sink at the rate the frame declares; a client
that pins a fixed rate plays the other engine's turns at the wrong speed and
pitch. The rate lives on the turn precisely so it can vary, and that will not be
replaced by a single server-wide constant.

**Unknown frame types and unknown fields are safely ignorable.** A client may
skip any text frame whose `type` it does not recognize, and ignore any field it
does not read, and keep working. This is the property that lets the protocol grow
additively - the entire turn-taking surface landed as new frame types
(`turn_started`) and new fields (`turn_id`, `reason`, `flush`) that pre-existing
clients simply ignored. You will never be forced to understand a newly added
frame or field just to keep using the features you already consume, so build your
receive loop to drop what it does not know rather than to error on it.

**The endpoint-hold upsert-by-index contract holds.** All frames for one
utterance share a single `segment_index`, and a client that treats that index as
a stable, revisable slot renders correctly across the interim -> provisional
endpoint -> confirmed final revision, with the speculative correction handled for
free (see [the upsert-by-index revision contract](#the-upsert-by-index-revision-contract)).
This is a durable part of the contract, not an incidental of the current STT
engine: the shared-index, revise-in-place shape is what you can depend on.

## Verifying against source

This doc names symbols, not line numbers, so it stays valid as the code moves. To
confirm any claim here against the code as it stands, or to catch drift after a
protocol change, diff these five files - together they define the entire
client-visible contract:

- `src/voicemanager/protocol.py` - every WebSocket message type and its fields,
  the close codes, and `AUDIO_FORMAT_SPEC`. The schema reference and message
  tables above are drawn from here.
- `src/voicemanager/transport/websocket.py` - the `/ws/stream` handler: auth
  before `accept()`, the handshake, audio validation, and the STT/TTS routing and
  turn-taking behaviour.
- `src/voicemanager/transport/http.py` - the REST surface: `/health`, `/info`,
  and `/voice/upload`.
- `src/voicemanager/auth/deps.py` - bearer token parsing and lookup for the
  WebSocket pre-accept check and the REST endpoints.
- `src/voicemanager/client.py` - the reference Python client, whose methods map
  onto the frames above and encode the client-side rules (post-cancel discard,
  turn stamping) as working code.

If a symbol named in this document is absent from these files, treat the document
as stale and fix it here - the source is authoritative.
