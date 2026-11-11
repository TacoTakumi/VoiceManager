"""The VM_LOG_LEVEL setting: the voicemanager loggers and the Tyger sidecar's flag."""
import logging
import sys

import pytest
from pydantic import ValidationError

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)

from voicemanager.cli import apply_log_level
from voicemanager.config import Settings
from voicemanager.pipeline.tyger.sidecar import TygerSidecar


def _settings(**overrides) -> Settings:
    base = dict(_env_file=None, turn_engine="tyger", device="cpu", tyger_py=sys.executable)
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def package_logger():
    logger = logging.getLogger("voicemanager")
    before = logger.level
    yield logger
    logger.setLevel(before)


def test_the_default_is_info_and_the_sidecar_gets_no_flag():
    settings = _settings()
    assert settings.log_level == "INFO"
    assert "--log-level" not in TygerSidecar.from_settings(settings, device="cpu").command()


def test_debug_reaches_the_package_loggers_and_not_the_root(package_logger):
    root_before = logging.getLogger().level
    apply_log_level(_settings(log_level="DEBUG").log_level)
    assert package_logger.level == logging.DEBUG
    assert logging.getLogger("voicemanager.transport.turn_engine").isEnabledFor(logging.DEBUG)
    assert logging.getLogger().level == root_before


def test_the_level_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("VM_LOG_LEVEL", "DEBUG")
    assert Settings(_env_file=None).log_level == "DEBUG"


def test_debug_is_passed_to_the_sidecar_as_a_flag_its_script_knows():
    cmd = TygerSidecar.from_settings(_settings(log_level="DEBUG"), device="cpu").command()
    assert cmd[cmd.index("--log-level") + 1] == "debug"
    assert '"--log-level"' in TygerSidecar.from_settings(
        _settings(), device="cpu").server_script.read_text()


def test_an_unknown_level_is_refused():
    with pytest.raises(ValidationError):
        _settings(log_level="LOUD")
