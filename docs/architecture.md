# Architecture

This page shows how VoiceManager is built: the processes, the path of one connection, and the turn engine. The WebSocket protocol that clients use is in [client-integration.md](client-integration.md).

## Processes

One FastAPI server (`python -m voicemanager` or `voicemanager serve`) holds the client connections. Engines that pin their own runtime run as sidecar processes that the server starts and stops.

```mermaid
flowchart LR
    client["Client<br/>any language<br/>voicemanager.client in Python"]
    browser["Browser"]

    subgraph server["VoiceManager server (FastAPI, Uvicorn)"]
        ws["/ws/stream<br/>transport/websocket.py"]
        rest["/health /info /voice/upload<br/>transport/http.py"]
        web["Web UI<br/>web/"]
        auth["Users and tokens<br/>auth/"]
        reg["SessionRegistry<br/>session.py"]
        inproc["In-process engines<br/>whisper, nemotron, qwen STT<br/>openvoice, vibevoice TTS"]
    end

    db[("SQLite user store<br/>VM_DB_PATH")]

    subgraph per["One pair per session"]
        sttsc["audio.cpp STT sidecar<br/>sidecar/ (C++)"]
        ttssc["audio.cpp TTS sidecar<br/>sidecar/ (C++)"]
    end

    voxtream["VoXtream sidecar<br/>pipeline/voxtream_server.py<br/>own venv, one process"]

    subgraph tyger["Tyger sidecar (venv-tyger, one process)"]
        tserver["pipeline/tyger/server.py<br/>scorer + turn rules"]
        voxlock["VoxLock pre-filter<br/>optional"]
    end

    client -- "WebSocket: PCM, JSON frames" --> ws
    client -- "HTTP, bearer token" --> rest
    browser -- "HTTP, sign-in session" --> web
    ws --> auth
    rest --> auth
    web --> auth
    auth --> db
    ws --> reg
    ws --> inproc
    ws -- "Unix socket" --> sttsc
    ws -- "Unix socket" --> ttssc
    ws -- "WebSocket per generation" --> voxtream
    ws -- "loopback WebSocket /tyger per session" --> tserver
    tserver --> voxlock
```

- The server reads `.env` (`VM_` prefix) once at startup. `VM_STT_ENGINE` and `VM_TTS_ENGINE` choose one engine of each kind for every connection.
- The lifespan in `app.py` loads the models and starts the sidecars the settings need. If a model or a sidecar fails to load, the server does not start.
- The audio.cpp engines start one sidecar process per session, so an engine crash stops only that session.
- VoXtream and Tyger pin their own PyTorch, so each runs as one process from its own virtual environment, and all sessions share it.
- All session state is in memory. It is lost when the server restarts.

## One connection

Binary frames carry mic audio to STT. JSON text frames carry TTS requests. STT and TTS run at the same time on the socket.

```mermaid
flowchart LR
    mic["Binary frames<br/>16 kHz mono PCM"] --> check["First-frame check<br/>format, WAV header"]
    check --> hold{"Turn engine<br/>opted in?"}
    hold -- "no" --> stt["STT pipeline"]
    hold -- "yes" --> mhold["Mic hold<br/>reply echo kept out"]
    mhold --> stt
    stt --> trans["transcript frames<br/>interim, final, turn_id"]

    text["tts, tts_stream,<br/>tts_cancel frames"] --> chunk["Sentence or clause chunker"]
    chunk --> tts["TTS pipeline"]
    tts --> pcm["audio_start, PCM,<br/>audio_end"]

    hold -- "yes" --> feed["Hop feed<br/>mic + played reply<br/>320 ms hops"]
    tts -. "copy of reply PCM" .-> feed
    edges["playback_start,<br/>playback_stop frames"] --> feed
    feed --> tyg["Tyger sidecar"]
    tyg --> tframes["turn_end, barge frames"]
    tyg -. "barge stops the reply" .-> tts
```

- A connection opts in to the turn engine with a `turn_engine` frame before its first audio. Only the audiocpp STT engine supports it. Code: `transport/turn_engine.py`.
- The client reports when each reply starts and stops playing. The hop feed uses these edges to put the reply's audio next to the mic audio it overlaps.
- A new TTS request cancels the request in flight. A barge from Tyger ends the reply on the server, with `audio_end` reason `interrupted`.

## Turn engine and the pre-filter

The Tyger sidecar scores each hop and applies the turn rules (`pipeline/tyger/rules.py`). A hop that overlaps a playing reply is in speaking mode. Any other hop is in listening mode.

With `VM_TYGER_PREFILTER=voxlock`, each session's scorer is wrapped in the pre-filter (`pipeline/tyger/prefilter.py`). It is experimental and off by default.

```mermaid
flowchart TD
    hop["Hop: mic + agent channel + mode"] --> enrolled{"Enrolled?"}
    enrolled -- "no" --> collect["In listening mode, keep frames<br/>above the voice gate"]
    collect --> full{"3 s of voice?"}
    full -- "yes" --> build["Build the VoxLock filter<br/>for this session"]
    collect --> plain["Tyger scores the raw hop"]
    enrolled -- "yes" --> filt["VoxLock filters the mic"]
    filt --> mode{"Speaking mode?"}
    mode -- "no" --> plain
    mode -- "yes" --> spk["Tyger scores the filtered mic<br/>+ noise at the floor<br/>+ silent agent channel"]
    plain --> rules["Turn rules: turn_end, barge"]
    spk --> rules
```

- The filter learns the user's voice from the first 3 s of speech heard while no reply plays.
- The STT always gets the raw mic. The pre-filter changes only what Tyger hears.
- VoxLock runs on the sidecar's GPU only. The sidecar exits at startup if it cannot run there.

## Source layout

| Path | Contents |
|---|---|
| `src/voicemanager/app.py` | App factory and lifespan: loads models, starts sidecars, wires `app.state` |
| `src/voicemanager/config.py` | `Settings`, read from `.env` |
| `src/voicemanager/protocol.py` | Pydantic WebSocket frames and the audio format |
| `src/voicemanager/session.py` | `SessionRegistry` of active sessions |
| `src/voicemanager/transport/` | HTTP and WebSocket handlers, turn ids, the turn engine link |
| `src/voicemanager/pipeline/` | STT and TTS pipelines per engine, the chunkers, the VoXtream sidecar |
| `src/voicemanager/pipeline/tyger/` | Tyger sidecar server, scorer, turn rules, pre-filter, vendored VoxLock |
| `src/voicemanager/models/` | Model loading and device selection |
| `src/voicemanager/auth/` | User store and bearer tokens |
| `src/voicemanager/web/` | Web UI for accounts and tokens |
| `src/voicemanager/client.py` | Python client library |
| `sidecar/` | audio.cpp sidecar (C++) and its protocol, `sidecar/PROTOCOL.md` |
