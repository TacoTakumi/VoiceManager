"""The wake module: phrase rules, the wake settings, the startup load and the sherpa spotter.

Model-free and server-free. sherpa-onnx and sentencepiece are optional installs and
this suite never imports them: the spotter class is driven through stand-ins for the
two packages, put in sys.modules for one test at a time. The stand-ins record what
the class hands the library, which matters because the real library ends the process
when it is handed a token it does not know or a keyword file it cannot read.
"""
import concurrent.futures
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import authkit  # test auth env; must be imported before voicemanager  # noqa: F401
from fastapi.testclient import TestClient

import voicemanager.app as app_module
from voicemanager import wake
from voicemanager.config import Settings, resolve_home
from voicemanager.wake import (
    MAX_PHRASE_CHARS,
    MAX_PHRASES,
    WakeModelError,
    WakePhraseError,
    find_model_files,
    load_wake_spotter,
    normalize_phrases,
    split_phrases,
)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --- phrase rules ----------------------------------------------------------------

def test_a_phrase_is_lowercased_and_its_spaces_collapsed():
    assert normalize_phrases(["  Hey   COMPUTER "]) == ["hey computer"]


def test_an_apostrophe_is_allowed():
    assert normalize_phrases(["don't panic"]) == ["don't panic"]


def test_a_repeated_phrase_is_kept_once_and_the_order_is_kept():
    assert normalize_phrases(["hey hermes", "Hey Computer", "HEY HERMES"]) == [
        "hey hermes", "hey computer"]


def test_an_empty_list_is_refused():
    with pytest.raises(WakePhraseError, match="empty"):
        normalize_phrases([])


@pytest.mark.parametrize("phrase", ["", "   ", "\t\n"])
def test_an_empty_phrase_is_refused(phrase):
    with pytest.raises(WakePhraseError, match="empty"):
        normalize_phrases(["hey computer", phrase])


@pytest.mark.parametrize("phrase", [
    "hey computer 9", "hey, computer", "héllo", "hey_computer", "hey-computer",
    "你好", "hey computer!", "a/b", "tab\x00",
])
def test_a_phrase_with_another_character_is_refused(phrase):
    with pytest.raises(WakePhraseError, match="not a letter"):
        normalize_phrases([phrase])


def test_the_most_phrases_are_accepted_and_one_more_is_refused():
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india"]
    assert len(normalize_phrases(names[:MAX_PHRASES])) == MAX_PHRASES == 8
    with pytest.raises(WakePhraseError, match="too many"):
        normalize_phrases(names[:MAX_PHRASES + 1])


def test_a_very_long_phrase_is_refused():
    assert normalize_phrases(["a" * MAX_PHRASE_CHARS]) == ["a" * MAX_PHRASE_CHARS]
    with pytest.raises(WakePhraseError, match="longer"):
        normalize_phrases(["a" * (MAX_PHRASE_CHARS + 1)])


def test_a_setting_value_is_split_at_commas():
    assert split_phrases("hey computer, hey hermes ,,") == ["hey computer", "hey hermes"]
    assert split_phrases("  ") == []


# --- settings --------------------------------------------------------------------

def test_wake_defaults():
    s = _settings()
    assert s.wake_engine == "none"
    assert s.wake_model_dir == ""
    assert s.wake_phrases == "hey computer"
    assert s.wake_threshold == 0.15
    assert s.wake_score == 1.5


def test_wake_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("VM_WAKE_ENGINE", "sherpa")
    monkeypatch.setenv("VM_WAKE_MODEL_DIR", "/models/kws")
    monkeypatch.setenv("VM_WAKE_PHRASES", "hey hermes, hey computer")
    monkeypatch.setenv("VM_WAKE_THRESHOLD", "0.25")
    monkeypatch.setenv("VM_WAKE_SCORE", "2")
    s = _settings()
    assert (s.wake_engine, s.wake_model_dir) == ("sherpa", "/models/kws")
    assert split_phrases(s.wake_phrases) == ["hey hermes", "hey computer"]
    assert (s.wake_threshold, s.wake_score) == (0.25, 2.0)


def test_a_relative_model_directory_resolves_against_the_home():
    assert _settings(wake_model_dir="models/kws").wake_model_dir == str(
        resolve_home() / "models" / "kws")


@pytest.mark.parametrize("field, value", [
    ("wake_engine", "porcupine"), ("wake_threshold", 0.0), ("wake_threshold", 1.5),
    ("wake_score", -1.0),
])
def test_out_of_range_wake_values_are_refused(field, value):
    with pytest.raises(ValidationError, match=field):
        _settings(**{field: value})


# --- the startup load ------------------------------------------------------------

@pytest.fixture
def no_optional_packages(monkeypatch):
    """Make an import of either optional package fail, as on a server without them."""
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    monkeypatch.setitem(sys.modules, "sentencepiece", None)


def _model_dir(tmp_path: Path, *, leave_out: str | None = None, int8: bool = False) -> Path:
    """A directory with the file names of an unpacked keyword spotting model."""
    names = ["encoder-epoch-12.onnx", "decoder-epoch-12.onnx", "joiner-epoch-12.onnx",
             "tokens.txt", "bpe.model"]
    if int8:
        names += ["encoder-epoch-12.int8.onnx", "decoder-epoch-12.int8.onnx",
                  "joiner-epoch-12.int8.onnx"]
    for name in names:
        if leave_out is None or not name.startswith(leave_out):
            (tmp_path / name).write_text("▁HE 0\nY 1\n" if name == "tokens.txt" else "x")
    return tmp_path


def test_engine_none_loads_nothing_and_needs_no_package(no_optional_packages):
    assert load_wake_spotter(_settings()) is None


def test_engine_sherpa_without_a_model_directory_names_the_setting(no_optional_packages):
    with pytest.raises(WakeModelError, match="VM_WAKE_MODEL_DIR"):
        load_wake_spotter(_settings(wake_engine="sherpa"))


def test_engine_sherpa_with_a_missing_directory_names_it(tmp_path, no_optional_packages):
    missing = tmp_path / "not-here"
    with pytest.raises(WakeModelError) as err:
        load_wake_spotter(_settings(wake_engine="sherpa", wake_model_dir=str(missing)))
    assert str(missing) in str(err.value)


def test_engine_sherpa_with_an_empty_directory_names_it(tmp_path, no_optional_packages):
    with pytest.raises(WakeModelError) as err:
        load_wake_spotter(_settings(wake_engine="sherpa", wake_model_dir=str(tmp_path)))
    assert str(tmp_path) in str(err.value)
    assert "encoder" in str(err.value)


@pytest.mark.parametrize("part", ["encoder", "decoder", "joiner", "tokens.txt", "bpe.model"])
def test_a_missing_model_file_is_named_with_the_directory(tmp_path, part, no_optional_packages):
    model = _model_dir(tmp_path, leave_out=part)
    with pytest.raises(WakeModelError) as err:
        load_wake_spotter(_settings(wake_engine="sherpa", wake_model_dir=str(model)))
    assert str(model) in str(err.value)
    assert part in str(err.value)


def test_a_missing_package_is_named_after_the_model_files_are_found(
        tmp_path, no_optional_packages):
    with pytest.raises(WakeModelError, match="pip install sherpa-onnx sentencepiece"):
        load_wake_spotter(
            _settings(wake_engine="sherpa", wake_model_dir=str(_model_dir(tmp_path))))


def test_a_bad_default_phrase_is_refused_before_the_model_is_read(no_optional_packages):
    with pytest.raises(WakeModelError, match="VM_WAKE_PHRASES"):
        load_wake_spotter(_settings(
            wake_engine="sherpa", wake_model_dir="/nowhere", wake_phrases="hey computer 2"))


def test_no_default_phrase_is_refused(no_optional_packages):
    with pytest.raises(WakeModelError, match="VM_WAKE_PHRASES"):
        load_wake_spotter(_settings(
            wake_engine="sherpa", wake_model_dir="/nowhere", wake_phrases=" , "))


def test_the_float_model_files_are_taken_over_the_int8_ones(tmp_path):
    files = find_model_files(str(_model_dir(tmp_path, int8=True)))
    assert not any(".int8." in files[part] for part in ("encoder", "decoder", "joiner"))


def test_int8_model_files_are_taken_when_there_are_no_float_ones(tmp_path):
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx",
                 "tokens.txt", "bpe.model"):
        (tmp_path / name).write_text("x")
    files = find_model_files(str(tmp_path))
    assert files["encoder"].endswith("encoder.int8.onnx")


# --- the sherpa spotter, on stand-ins for the two packages ------------------------

# The stand-in model's tokens: one per upper-case letter and the apostrophe, plus the
# word-start mark, as tokens.txt lists them.
_VOCAB = ["▁"] + [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["'"]


class _FakeTokenizer:
    """Cuts text into one token per character, with a word-start mark per word."""

    def __init__(self, model_file: str) -> None:
        self.model_file = model_file

    def encode(self, text: str, out_type=str) -> list[str]:
        tokens: list[str] = []
        for word in text.split():
            tokens.append("▁")
            tokens.extend(word)
        return tokens


class _FakeStream:
    def __init__(self, keywords: str) -> None:
        self.keywords = keywords
        self.accepted: list[np.ndarray] = []
        self.resets = 0

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        assert sample_rate == 16000
        self.accepted.append(samples)


class _FakeKeywordSpotter:
    """Records how it was built and reports the hits a test queues.

    ``hits`` holds one entry per decode step: a tag, or "" for no hit. A stream is
    ready for one decode step per accepted frame.
    """

    made: list["_FakeKeywordSpotter"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        # The real library reads the keyword file here and not after.
        self.keyword_file_text = Path(kwargs["keywords_file"]).read_text(encoding="utf-8")
        self.streams: list[_FakeStream] = []
        self.hits: list[str] = []
        self._decoded: dict[int, int] = {}
        self._result = ""
        _FakeKeywordSpotter.made.append(self)

    def create_stream(self, keywords: str) -> _FakeStream:
        self.streams.append(_FakeStream(keywords))
        return self.streams[-1]

    def is_ready(self, stream: _FakeStream) -> bool:
        return self._decoded.get(id(stream), 0) < len(stream.accepted)

    def decode_stream(self, stream: _FakeStream) -> None:
        self._decoded[id(stream)] = self._decoded.get(id(stream), 0) + 1
        self._result = self.hits.pop(0) if self.hits else ""

    def get_result(self, stream: _FakeStream) -> str:
        return self._result

    def reset_stream(self, stream: _FakeStream) -> None:
        stream.resets += 1
        self._result = ""


@pytest.fixture
def sherpa(monkeypatch, tmp_path):
    """Stand-ins for sherpa_onnx and sentencepiece, and a model directory for them."""
    _FakeKeywordSpotter.made = []
    monkeypatch.setitem(
        sys.modules, "sherpa_onnx", types.SimpleNamespace(KeywordSpotter=_FakeKeywordSpotter))
    monkeypatch.setitem(
        sys.modules, "sentencepiece",
        types.SimpleNamespace(SentencePieceProcessor=_FakeTokenizer))
    model = _model_dir(tmp_path)
    (model / "tokens.txt").write_text(
        "".join(f"{tok} {i}\n" for i, tok in enumerate(_VOCAB)), encoding="utf-8")
    return model


def _spotter(model: Path, phrases=("hey computer",), threshold=0.15, score=1.5):
    return wake.SherpaWakeSpotter(
        str(model), default_phrases=list(phrases), threshold=threshold, score=score)


def test_the_loader_builds_the_spotter_from_the_settings(sherpa):
    spotter = load_wake_spotter(_settings(
        wake_engine="sherpa", wake_model_dir=str(sherpa),
        wake_phrases="Hey Computer, hey hermes", wake_threshold=0.25, wake_score=2.0))
    try:
        assert spotter.default_phrases == ["hey computer", "hey hermes"]
        built = _FakeKeywordSpotter.made[0].kwargs
        assert built["keywords_threshold"] == 0.25
        assert built["keywords_score"] == 2.0
        assert built["provider"] == "cpu"
        assert built["tokens"] == str(sherpa / "tokens.txt")
        assert built["encoder"] == str(sherpa / "encoder-epoch-12.onnx")
    finally:
        spotter.close()


def test_the_spotter_is_built_with_one_keyword_that_speech_cannot_match(sherpa):
    spotter = _spotter(sherpa)
    try:
        lines = _FakeKeywordSpotter.made[0].keyword_file_text.splitlines()
        assert len(lines) == 1
        fields = lines[0].split()
        # A threshold of 1.0 is a probability no decode reaches, and the keyword is
        # made of the model's own tokens, which the library needs.
        assert "#1.0" in fields
        assert "@PLACEHOLDER" in fields
        tokens = [f for f in fields if f[0] not in ":#@"]
        assert tokens and all(tok in _VOCAB for tok in tokens)
        # None of the connection's phrases is in the shared file.
        assert "H E Y" not in lines[0]
    finally:
        spotter.close()


def test_the_keyword_file_is_gone_after_the_spotter_is_built(sherpa):
    spotter = _spotter(sherpa)
    try:
        assert not Path(_FakeKeywordSpotter.made[0].kwargs["keywords_file"]).exists()
    finally:
        spotter.close()


def test_a_stream_gets_its_phrases_as_model_tokens_with_one_tag_each(sherpa):
    spotter = _spotter(sherpa)
    try:
        stream = spotter.open_stream(["hey computer", "don't panic"])
        assert stream.phrases == ["hey computer", "don't panic"]
        assert _FakeKeywordSpotter.made[0].streams[0].keywords == (
            "▁ H E Y ▁ C O M P U T E R @W0/▁ D O N ' T ▁ P A N I C @W1")
    finally:
        spotter.close()


def test_two_streams_have_separate_phrase_lists(sherpa):
    spotter = _spotter(sherpa)
    try:
        spotter.open_stream(["hey hermes"])
        spotter.open_stream(spotter.default_phrases)
        first, second = _FakeKeywordSpotter.made[0].streams
        assert "H E R M E S" in first.keywords and "C O M P U T E R" not in first.keywords
        assert "C O M P U T E R" in second.keywords and "H E R M E S" not in second.keywords
    finally:
        spotter.close()


def test_a_phrase_the_model_has_no_tokens_for_never_reaches_the_library(sherpa):
    # The stand-in tokenizer gives one token per character, so a digit is a token
    # tokens.txt does not have. The real library ends the process on such a token.
    spotter = _spotter(sherpa)
    try:
        with pytest.raises(WakePhraseError, match="no tokens"):
            spotter.open_stream(["hey computer", "route 66"])
        assert _FakeKeywordSpotter.made[0].streams == []
    finally:
        spotter.close()


def test_a_default_phrase_the_model_has_no_tokens_for_stops_the_load(sherpa):
    (sherpa / "tokens.txt").write_text("▁ 0\nH 1\nE 2\nY 3\nQ 4\n", encoding="utf-8")
    with pytest.raises(WakeModelError, match="VM_WAKE_PHRASES"):
        load_wake_spotter(_settings(
            wake_engine="sherpa", wake_model_dir=str(sherpa), wake_phrases="hey computer"))


def test_feed_gives_the_library_float_samples_and_reports_no_phrase(sherpa):
    spotter = _spotter(sherpa)
    try:
        stream = spotter.open_stream(["hey computer"])
        pcm = np.array([0, 16384, -32768, 32767], dtype="<i2").tobytes()
        assert stream.feed(pcm) is None
        got = _FakeKeywordSpotter.made[0].streams[0].accepted[0]
        assert got.dtype == np.float32
        np.testing.assert_allclose(got, [0.0, 0.5, -1.0, 32767 / 32768])
    finally:
        spotter.close()


def test_feed_reports_the_phrase_of_the_tag_and_resets_the_stream(sherpa):
    spotter = _spotter(sherpa)
    try:
        stream = spotter.open_stream(["hey computer", "hey hermes"])
        fake = _FakeKeywordSpotter.made[0]
        fake.hits = ["", "W1"]
        assert stream.feed(b"\x00\x00" * 160) is None
        assert stream.feed(b"\x00\x00" * 160) == "hey hermes"
        assert fake.streams[0].resets == 1
        # After the reset the stream goes on and can report again.
        fake.hits = ["W0"]
        assert stream.feed(b"\x00\x00" * 160) == "hey computer"
        assert fake.streams[0].resets == 2
    finally:
        spotter.close()


def test_a_hit_of_the_placeholder_is_dropped_and_the_stream_is_reset(sherpa):
    spotter = _spotter(sherpa)
    try:
        stream = spotter.open_stream(["hey computer"])
        fake = _FakeKeywordSpotter.made[0]
        fake.hits = ["PLACEHOLDER"]
        assert stream.feed(b"\x00\x00" * 160) is None
        assert fake.streams[0].resets == 1
    finally:
        spotter.close()


def test_an_odd_byte_is_carried_to_the_next_frame(sherpa):
    spotter = _spotter(sherpa)
    try:
        stream = spotter.open_stream(["hey computer"])
        samples = np.array([100, 200, 300], dtype="<i2").tobytes()
        assert stream.feed(samples[:3]) is None
        assert stream.feed(samples[3:]) is None
        accepted = _FakeKeywordSpotter.made[0].streams[0].accepted
        np.testing.assert_allclose(
            np.concatenate(accepted) * 32768.0, [100, 200, 300], rtol=0, atol=1e-3)
    finally:
        spotter.close()


def test_a_frame_of_one_byte_is_held_and_nothing_is_fed(sherpa):
    spotter = _spotter(sherpa)
    try:
        stream = spotter.open_stream(["hey computer"])
        assert stream.feed(b"\x01") is None
        assert stream.feed(b"") is None
        assert _FakeKeywordSpotter.made[0].streams[0].accepted == []
    finally:
        spotter.close()


def test_the_spotter_has_one_worker_thread_and_close_stops_it(sherpa):
    spotter = _spotter(sherpa)
    assert spotter.executor._max_workers == 1
    assert spotter.executor.submit(lambda: 7).result(timeout=5) == 7
    spotter.close()
    assert spotter.executor is None
    spotter.close()  # a second close does nothing


# --- the app keeps the spotter ---------------------------------------------------

def _stub_engines(monkeypatch, settings: Settings) -> None:
    """Run the real lifespan with stand-in STT and TTS engines."""
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(app_module, "load_models", lambda s: {
        "stt_engine_type": "qwen", "device": "cpu", "qwen_model": object()})
    monkeypatch.setattr(app_module, "load_tts_models", lambda s: {
        "tts_engine": object(),
        "tts_executor": concurrent.futures.ThreadPoolExecutor(max_workers=1),
        "tts_engine_type": "openvoice"})


def test_the_default_server_has_no_spotter(monkeypatch, no_optional_packages):
    _stub_engines(monkeypatch, _settings())
    with TestClient(app_module.create_app()) as client:
        assert client.app.state.wake_spotter is None
        assert client.get("/health").status_code == 200


def test_the_app_keeps_the_loaded_spotter_and_closes_it_at_shutdown(monkeypatch, sherpa):
    _stub_engines(monkeypatch, _settings(
        wake_engine="sherpa", wake_model_dir=str(sherpa), wake_phrases="hey hermes"))
    with TestClient(app_module.create_app()) as client:
        spotter = client.app.state.wake_spotter
        assert isinstance(spotter, wake.SherpaWakeSpotter)
        assert spotter.default_phrases == ["hey hermes"]
        assert spotter.executor is not None
    assert spotter.executor is None


def test_a_wake_engine_that_cannot_start_stops_the_server_before_the_large_models(
        monkeypatch, tmp_path, no_optional_packages):
    _stub_engines(monkeypatch, _settings(wake_engine="sherpa", wake_model_dir=str(tmp_path)))

    def load_models(settings):
        raise AssertionError("the large models must not load after a wake failure")

    monkeypatch.setattr(app_module, "load_models", load_models)
    with pytest.raises(WakeModelError) as err:
        with TestClient(app_module.create_app()):
            pass
    assert str(tmp_path) in str(err.value)


