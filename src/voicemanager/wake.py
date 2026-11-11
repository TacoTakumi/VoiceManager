"""Server-side wake word: phrase rules, the spotter interface and the sherpa-onnx spotter.

A client with no keyword spotter of its own can ask the server to listen for a wake
phrase. This module holds everything about the phrases and the spotter; the WebSocket
handler holds the connection state (see transport/websocket.py).

The engine is the sherpa-onnx keyword spotter (a small streaming zipformer
transducer). Its phrases are plain text: the text is cut into the model's tokens and
the decoder is boosted toward that token sequence, so a new phrase needs no training.
One spotter is loaded at startup and shared; each connection gets its own stream,
with its own phrase list.

sherpa-onnx and sentencepiece are optional installs. They are imported only when
VM_WAKE_ENGINE=sherpa, inside the functions that need them, so the server and the
test suite start without them.

Two properties of sherpa-onnx shape this code:

* It ends the process, with no Python exception, when it is given a keyword it
  cannot encode (a token that is not in tokens.txt) or a keyword file it cannot
  read. So every token is checked against tokens.txt here before it reaches the
  library, and the model files are checked before the spotter is built.
* A stream's keywords are added to the keywords of the spotter's own file; they do
  not replace them. So the shared spotter is built with one placeholder keyword that
  speech cannot match, and every phrase a connection listens for is a stream keyword.
"""
from __future__ import annotations

import logging
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

# The most phrases one connection can listen for. Every phrase adds a path the
# decoder is boosted toward, and with it a chance of a false wake.
MAX_PHRASES = 8
# A wake phrase is a few words. The cap keeps a client from sending a paragraph.
MAX_PHRASE_CHARS = 64

# Letters, single spaces between words and apostrophes ("don't"). Digits and
# punctuation have no spoken form the model could match.
_PHRASE_RE = re.compile(r"[a-z']+( [a-z']+)*")

# The placeholder keyword's text and its tag. Its threshold of 1.0 is a probability
# no decode reaches, and its tag is not a phrase tag, so a match would be dropped too.
_PLACEHOLDER_TEXT = "QQQQQQQQ"
_PLACEHOLDER_TAG = "PLACEHOLDER"


class WakePhraseError(ValueError):
    """A phrase list the server cannot use. The message is safe to send to a client."""


class WakeModelError(RuntimeError):
    """The wake engine cannot start. The message names the setting or path at fault."""


def normalize_phrases(phrases: list[str]) -> list[str]:
    """Check a phrase list and return it in the form the spotter uses.

    Each phrase is lowercased and its runs of whitespace become one space. A phrase
    that is then repeated is kept once. Raises WakePhraseError for an empty list, an
    empty phrase, a phrase with a character other than a letter, a space or an
    apostrophe, a phrase longer than MAX_PHRASE_CHARS, and more than MAX_PHRASES
    phrases.
    """
    if not phrases:
        raise WakePhraseError("the phrase list is empty")
    if len(phrases) > MAX_PHRASES:
        raise WakePhraseError(
            f"too many phrases: {len(phrases)}, the most a connection can have is {MAX_PHRASES}")
    out: list[str] = []
    for raw in phrases:
        phrase = " ".join(raw.lower().split())
        if not phrase:
            raise WakePhraseError("a phrase is empty")
        if len(phrase) > MAX_PHRASE_CHARS:
            raise WakePhraseError(
                f"a phrase is longer than {MAX_PHRASE_CHARS} characters: {phrase[:20]!r}...")
        if not _PHRASE_RE.fullmatch(phrase):
            raise WakePhraseError(
                f"the phrase {phrase!r} has a character that is not a letter, "
                "a space or an apostrophe")
        if phrase not in out:
            out.append(phrase)
    return out


def split_phrases(value: str) -> list[str]:
    """The phrases of a comma-separated setting value, blank entries left out."""
    return [part.strip() for part in value.split(",") if part.strip()]


class WakeStream(Protocol):
    """One connection's spotter stream."""

    phrases: list[str]

    def feed(self, pcm: bytes) -> str | None:
        """Take 16 kHz mono 16-bit PCM and return the phrase heard in it, or None."""


class WakeSpotter(Protocol):
    """What the WebSocket handler needs from a wake engine.

    ``executor`` is where the handler runs ``open_stream`` and a stream's ``feed``
    (both block on the model); None means the event loop's default executor.
    """

    default_phrases: list[str]
    executor: ThreadPoolExecutor | None

    def open_stream(self, phrases: list[str]) -> WakeStream:
        """A new stream that listens for ``phrases`` (as normalize_phrases gives them).

        Raises WakePhraseError for a phrase the model has no tokens for.
        """

    def close(self) -> None:
        """Release the engine at shutdown."""


def find_model_files(model_dir: str) -> dict[str, str]:
    """The model files of a sherpa-onnx keyword spotting model directory.

    The directory is the unpacked model archive: an encoder, a decoder and a joiner
    (.onnx), tokens.txt and bpe.model. The float files are taken when the directory
    has both float and int8 files. Raises WakeModelError, naming the directory and
    the file, when one is missing.
    """
    root = Path(model_dir).expanduser()
    if not root.is_dir():
        raise WakeModelError(
            f"the wake model directory {str(root)!r} does not exist "
            "(VM_WAKE_MODEL_DIR; download it with `python setup_models.py wake`)")
    found: dict[str, str] = {}
    for part in ("encoder", "decoder", "joiner"):
        files = sorted(root.glob(f"{part}*.onnx"))
        if not files:
            raise WakeModelError(
                f"the wake model directory {str(root)!r} has no {part}*.onnx file "
                "(VM_WAKE_MODEL_DIR)")
        floats = [f for f in files if ".int8." not in f.name]
        found[part] = str((floats or files)[0])
    for name in ("tokens.txt", "bpe.model"):
        path = root / name
        if not path.is_file():
            raise WakeModelError(
                f"the wake model directory {str(root)!r} has no {name} (VM_WAKE_MODEL_DIR)")
        found[name] = str(path)
    return found


class SherpaWakeStream:
    """One connection's stream on the shared sherpa-onnx spotter."""

    def __init__(self, spotter, stream, phrases: list[str], by_tag: dict[str, str]) -> None:
        self.phrases = phrases
        self._spotter = spotter
        self._stream = stream
        self._by_tag = by_tag
        # The odd last byte of a frame, kept for the next one: a sample is two bytes.
        self._carry = b""

    def feed(self, pcm: bytes) -> str | None:
        if self._carry:
            pcm = self._carry + pcm
            self._carry = b""
        if len(pcm) % 2:
            self._carry = pcm[-1:]
            pcm = pcm[:-1]
        if not pcm:
            return None
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self._stream.accept_waveform(SAMPLE_RATE, samples)
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            tag = self._spotter.get_result(self._stream)
            if tag:
                # The library keeps reporting a hit until the stream is reset.
                self._spotter.reset_stream(self._stream)
                phrase = self._by_tag.get(tag)
                if phrase is not None:
                    return phrase
        return None


class SherpaWakeSpotter:
    """The sherpa-onnx keyword spotter, loaded once and shared by all connections.

    All model calls run on one worker thread (``executor``): the library's thread
    safety across streams is not documented, and one thread serves many streams (a
    stream costs a few percent of one core).
    """

    def __init__(self, model_dir: str, *, default_phrases: list[str],
                 threshold: float, score: float) -> None:
        files = find_model_files(model_dir)
        try:
            import sentencepiece
            import sherpa_onnx
        except ImportError as exc:
            raise WakeModelError(
                "VM_WAKE_ENGINE=sherpa needs the sherpa-onnx and sentencepiece packages "
                f"(pip install sherpa-onnx sentencepiece): {exc}") from exc

        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=files["bpe.model"])
        with open(files["tokens.txt"], encoding="utf-8") as fh:
            self._vocab = {line.split()[0] for line in fh if line.strip()}

        placeholder = self._tokens(_PLACEHOLDER_TEXT)
        # Tokenize the default list now, so a default phrase the model cannot take
        # stops the server at startup and not at a client's opt-in.
        self._keywords(default_phrases)
        self.default_phrases = default_phrases
        # The library reads the keyword file while it builds the spotter and not after.
        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as kw:
            kw.write(f"{' '.join(placeholder)} :0.0 #1.0 @{_PLACEHOLDER_TAG}\n")
            kw.flush()
            self._spotter = sherpa_onnx.KeywordSpotter(
                encoder=files["encoder"],
                decoder=files["decoder"],
                joiner=files["joiner"],
                tokens=files["tokens.txt"],
                keywords_file=kw.name,
                keywords_score=score,
                keywords_threshold=threshold,
                num_threads=1,
                provider="cpu",
            )
        self.executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="wake")
        logger.info("Wake spotter loaded: model=%s phrases=%s threshold=%s score=%s",
                    model_dir, default_phrases, threshold, score)

    def _tokens(self, phrase: str) -> list[str]:
        """The model tokens of a phrase. The model's text is upper case."""
        tokens = self._tokenizer.encode(phrase.upper(), out_type=str)
        missing = [t for t in tokens if t not in self._vocab]
        if not tokens or missing:
            raise WakePhraseError(f"the wake model has no tokens for the phrase {phrase!r}")
        return tokens

    def _keywords(self, phrases: list[str]) -> tuple[str, dict[str, str]]:
        """The stream keyword text for a phrase list, and the phrase of each tag."""
        entries = []
        by_tag: dict[str, str] = {}
        for index, phrase in enumerate(phrases):
            tag = f"W{index}"
            entries.append(f"{' '.join(self._tokens(phrase))} @{tag}")
            by_tag[tag] = phrase
        return "/".join(entries), by_tag

    def open_stream(self, phrases: list[str]) -> SherpaWakeStream:
        keywords, by_tag = self._keywords(phrases)
        return SherpaWakeStream(
            self._spotter, self._spotter.create_stream(keywords), list(phrases), by_tag)

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None


def load_wake_spotter(settings) -> WakeSpotter | None:
    """Load the wake engine at startup, or nothing when it is off.

    With VM_WAKE_ENGINE=none (the default) nothing is imported and None comes back.
    With 'sherpa' the model in VM_WAKE_MODEL_DIR is loaded; a blank or missing
    directory, a missing model file, a missing package and a default phrase the
    model cannot take each raise WakeModelError, which the lifespan turns into a
    refusal to start.
    """
    engine = settings.wake_engine
    if engine == "none":
        return None
    if engine != "sherpa":
        raise WakeModelError(f"unknown VM_WAKE_ENGINE {engine!r}: expected 'none' or 'sherpa'")
    if not settings.wake_model_dir:
        raise WakeModelError(
            "VM_WAKE_ENGINE=sherpa needs VM_WAKE_MODEL_DIR, the directory of the "
            "keyword spotting model (download it with `python setup_models.py wake`)")
    try:
        default_phrases = normalize_phrases(split_phrases(settings.wake_phrases))
        return SherpaWakeSpotter(
            settings.wake_model_dir,
            default_phrases=default_phrases,
            threshold=settings.wake_threshold,
            score=settings.wake_score,
        )
    except WakePhraseError as exc:
        raise WakeModelError(f"VM_WAKE_PHRASES cannot be used: {exc}") from exc
