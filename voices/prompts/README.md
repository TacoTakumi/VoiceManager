# Designing voices

You describe a voice in plain English, the engine invents it, you listen to a few
takes and keep the one you like. No recording, no voice actor, no audio editing.

Everything here drives `scripts/design_voice.py`, which talks to the Breeze-TTS
engine directly. It needs no running VoiceManager server.

## How it works

Breeze makes a voice in two steps, and the script runs both for you.

1. **Design.** Your written description goes to the engine as an instruction,
   with no reference audio at all. The engine invents a speaker matching the
   description and has it read your reference line. That recording *is* the new
   voice.
2. **Audition.** The engine then clones the clip from step one and reads your
   audition text with it. This is the file to judge, because the server clones
   exactly the same way when it speaks.

A registered voice is the step one clip plus its exact transcript. That pair is
all Breeze needs, and it is why the transcript is mandatory - the engine has to
know what the reference audio says.

## Seeds

Voice design is random. The description narrows the field, the seed picks one
voice out of it. Same description at three seeds gives three different people,
all matching the description.

That is the only difference between takes. Nothing else changes.

- If every take is wrong in the same way, rewrite the description.
- If they are the right sort of person but none is the one, ask for more seeds.

Once you pick a take, the audio file is the asset. The seed is recorded as
provenance only, because there is no promise a seed reproduces the same voice on
a different engine build or quantization.

## File format

One file per voice: markdown with a small YAML header. Two sections are
required.

### Header keys

| Key | Required | Meaning |
| --- | --- | --- |
| `name` | no | Preset name the server will use. Defaults to the filename. |
| `seeds` | no | Which takes to generate. Defaults to `[101]`. `--seeds` overrides it. |
| `options` | no | Extra engine knobs, such as `temperature: 0.8`. Rarely needed. |

### Sections

- **`# Voice`** (required) is the description, and the only writing that really
  matters. Describe the person, not the words: age, sex, pitch, pace, texture,
  accent, and the room they sound like they are in. Two or three sentences beat
  a paragraph.
- **`# Reference line`** (required) is what the new voice says to create itself.
  Aim for 8 to 10 seconds, roughly 25 to 35 words. A slow voice will run longer,
  which is fine.
- **`# Audition`** (optional) is the passage you judge the voice on. Make it look
  like the real job. A default is used if you leave it out.
- **`# Notes`** (optional) is ignored by the script. Write anything.

## Example: a starship captain

`captain.md`, generated at seeds 311, 312 and 313.

```markdown
---
name: designed_captain_f01
seeds: [311, 312, 313]
---

# Voice

A woman in her forties with a low, warm, slightly gravelly voice. She speaks
unhurried and lands the ends of her sentences, the way someone does when nobody
has interrupted her in twenty years. Dry, composed, faintly amused. Close-miked
in a quiet room, no reverb.

# Reference line

This is the captain. All decks report ready for departure, and the yard has
cleared us for the outer marker. Take us out on my mark, helm, ahead one quarter.

# Audition

Captain's log, stardate four seven six three point nine. We have entered the
Veridian system after a six day crossing at warp seven...
```

Note what the description does. "Lands the ends of her sentences" and "nobody has
interrupted her in twenty years" are about delivery and bearing, which the engine
can act on. The reference line is also in character, which helps.

## Example: a game master

`gamemaster.md`, for the hands-free voice RPG in `../LLMaster`, generated at
seeds 401, 402 and 403.

```markdown
---
name: designed_gm_01
seeds: [401, 402, 403]
---

# Voice

An old man with a dry, raspy voice, worn thin at the edges but still carrying.
He speaks slowly and leaves room between his sentences, in no hurry at all,
because he already knows how the story goes. Low and close, gravel under it,
patient rather than solemn. Close-miked in a small quiet room, no reverb.

# Reference line

The road bends south around the shoulder of the hill, and by the time you reach
the treeline the light has gone amber. Somewhere below you, water is moving over
stone.

# Audition

The door gives on the third push, and the smell that comes out is old paper and
colder air. Your lamp finds the edge of a table, then a chair pushed back as
though someone stood up quickly and never sat down again. Along the far wall,
shelves run the full height of the room. Most are empty. What do you do?
```

The audition text is second person, present tense, and ends on a question,
because that is what a GM actually does. Judge a narration voice on narration,
not on one line.

The slow cadence asked for here made the reference clips longer than the captain
ones, 9.8 to 13.0 seconds against 8.9 to 10.9. Expected, and still in range.

## Running it

Generate the takes named in the file:

    ./venv/bin/python scripts/design_voice.py voices/prompts/captain.md

Or choose your own seeds:

    ./venv/bin/python scripts/design_voice.py voices/prompts/captain.md --seeds 7,8,9

Takes land in `voice_design/<name>/`, two files per seed:

- `<name>_s<seed>.wav` is the reference clip, the voice itself.
- `<name>_s<seed>_audition.wav` is that voice reading the audition text. **Listen
  to this one.** It is how the server will sound.

A `.log` per leg sits beside them if a take fails.

## Keeping one

    ./venv/bin/python scripts/design_voice.py voices/prompts/captain.md --register --seed 312

That copies the reference clip into `voices/designed/` and adds an entry to
`refs.json` with the transcript and a provenance note. Restart the server and the
name works in the `voice` field of any `tts` or `tts_stream` request.

Nothing is written outside `voice_design/` until you pass `--register`, so
generating takes is always safe to repeat. Registering refuses to overwrite an
existing preset or WAV; rename the voice in the prompt file, or remove the old
entry by hand if you mean to replace it.

Startup validates every preset and fails loudly on a missing file or a missing
transcript, so a bad registration will not sneak into a running server.

## Practical notes

- Budget about a minute per seed. Each take loads a 5 GB model twice.
- Describe delivery, not content. "Unhurried, faintly amused, speaks in complete
  sentences" shapes the voice. "Says wise things" does nothing.
- Age, sex, texture and pace read reliably. Impressions of specific real people
  do not.
- A reference clip much under 6 seconds or over 14 clones poorly. The script
  warns when a take falls outside that range. Adjust the reference line length
  and run again.
- Output is 24 kHz mono, matching what the server serves under Breeze.
- The GPU used is `VM_AUDIOCPP_DEVICE` from `.env`. Override with `--gpu`.
