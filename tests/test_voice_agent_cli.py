"""The reference voice agent's command line: three subcommands, the options each takes.

Model-free. The script is loaded by path, as the micro-turn tests load their module.
Covers: the classic, microturn and replay subcommands parse with their defaults; the
micro-turn options and run-file defaults are the ones the micro-turn mode always had;
running the micro-turn module directly points at this command and exits 2.
"""
import importlib.util
import pathlib
import subprocess
import sys

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location("voice_agent", SCRIPTS / "voice_agent.py")
va = importlib.util.module_from_spec(_spec)
sys.modules["voice_agent"] = va
_spec.loader.exec_module(va)


def test_classic_defaults():
    args = va.build_parser().parse_args(["classic"])
    assert args.mode == "classic"
    assert args.vm_host == "localhost" and args.vm_port == 8255
    assert args.url == "http://127.0.0.1:8080" and args.model is None
    assert args.tts_mode == "clause" and args.voice is None and args.duration is None
    assert args.log is None and args.wav is None


def test_classic_options():
    args = va.build_parser().parse_args(
        ["classic", "--model", "gemma-4-12b", "--tts-mode", "sentence", "--duration", "30"])
    assert args.model == "gemma-4-12b" and args.tts_mode == "sentence"
    assert args.duration == 30.0
    with pytest.raises(SystemExit):
        va.build_parser().parse_args(["classic", "--tts-mode", "word"])


def test_microturn_keeps_the_live_mode_options():
    args = va.build_parser().parse_args(["microturn", "--duration", "5", "--voice", "alice"])
    assert args.mode == "microturn"
    assert args.vm_host == "localhost" and args.vm_port == 8255
    assert args.duration == 5.0 and args.voice == "alice"
    assert args.log is None and args.wav is None
    assert args.url == "http://127.0.0.1:8080"
    assert va.build_parser().parse_args(["microturn"]).tts_mode == "clause"
    assert va.build_parser().parse_args(["microturn", "--tts-mode", "flush"]).tts_mode == "flush"
    with pytest.raises(SystemExit):
        va.build_parser().parse_args(["microturn", "--tts-mode", "word"])


def test_replay_options(tmp_path):
    script = tmp_path / "call.txt"
    script.write_text("hello\n")
    args = va.build_parser().parse_args(["replay", "--script", str(script)])
    assert args.mode == "replay" and args.script == script
    assert va.build_parser().parse_args(["replay"]).script is None


def test_a_mode_is_required():
    with pytest.raises(SystemExit):
        va.build_parser().parse_args([])


def test_microturn_module_run_directly_points_at_the_command():
    result = subprocess.run([sys.executable, str(SCRIPTS / "microturn_client.py"), "replay"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2
    assert "voice_agent.py" in result.stderr
