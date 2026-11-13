"""The pre-filter manual check script runs at the values the sidecar ships with.

The script needs the GPU, the sidecar environment and captured calls, so it is not
run here: it is only imported (its ``main`` stays behind ``__name__ == "__main__"``).
The floor, the voice gate, the Tyger model and revision and the barge threshold it
uses must equal the Settings field defaults and the TurnRuleSettings defaults, and
the script must not carry its own copies of those values.
"""
import ast
import importlib.util
from pathlib import Path

import pytest

from voicemanager.config import Settings
from voicemanager.pipeline.tyger.rules import TurnRuleSettings

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tyger_prefilter_check.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("tyger_prefilter_check_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default(name):
    return Settings.model_fields[name].default


def test_floor_is_the_settings_default(script):
    assert script.FLOOR_DB == _default("tyger_prefilter_floor_db")


def test_voice_gate_is_the_settings_default(script):
    assert script.GATE_DB == _default("tyger_voice_gate_db")


def test_tyger_model_and_revision_are_the_settings_defaults(script):
    assert script.TYGER_MODEL == _default("tyger_model")
    assert script.TYGER_REVISION == _default("tyger_revision")


def test_barge_threshold_is_the_turn_rule_default(script):
    assert script.BARGE_THRESHOLD == TurnRuleSettings().barge_threshold


def test_the_script_holds_no_copy_of_the_shipped_values():
    source = SCRIPT.read_text()
    assert _default("tyger_revision") not in source
    numbers = {
        abs(node.value) for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and type(node.value) in (int, float)
    }
    for value in (_default("tyger_prefilter_floor_db"), _default("tyger_voice_gate_db"),
                  TurnRuleSettings().barge_threshold):
        assert abs(value) not in numbers, f"the script has its own copy of {value}"
