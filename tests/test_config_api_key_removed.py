"""The shared-key setting is gone: settings load without it and refuse it when set.

Model-free and server-free. The removed names come from authkit so no collected
test file spells them.
"""
import pytest

import authkit

import voicemanager.config as cfg
from voicemanager.config import RemovedSettingError, Settings

TOKEN_HINT = "voicemanager user token add"


@pytest.fixture
def no_legacy_key(monkeypatch):
    monkeypatch.delenv(authkit.LEGACY_KEY_ENV, raising=False)


def _assert_names_key_and_hint(message: str) -> None:
    assert authkit.LEGACY_KEY_ENV in message
    assert TOKEN_HINT in message
    assert "\n" not in message.strip(), "the message must be one line"


def test_settings_load_without_the_key(no_legacy_key):
    settings = Settings(_env_file=None)

    assert not hasattr(settings, authkit.LEGACY_KEY_FIELD)
    assert authkit.LEGACY_KEY_FIELD not in Settings.model_fields


def test_key_in_the_environment_refuses_to_load(no_legacy_key, monkeypatch):
    monkeypatch.setenv(authkit.LEGACY_KEY_ENV, "old-shared-key")

    with pytest.raises(RemovedSettingError) as exc:
        Settings(_env_file=None)

    _assert_names_key_and_hint(str(exc.value))


def test_key_in_an_env_file_refuses_to_load(no_legacy_key, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"VM_HOST=127.0.0.1\n{authkit.LEGACY_KEY_ENV}=old-shared-key\n")

    with pytest.raises(RemovedSettingError) as exc:
        Settings(_env_file=env_file)

    _assert_names_key_and_hint(str(exc.value))


def test_key_in_the_default_env_file_refuses_to_load(no_legacy_key, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"{authkit.LEGACY_KEY_ENV}=\n")
    monkeypatch.setenv(cfg.HOME_ENV, str(tmp_path))  # the default .env is <home>/.env

    with pytest.raises(RemovedSettingError) as exc:
        Settings()

    _assert_names_key_and_hint(str(exc.value))


def test_env_file_without_the_key_loads(no_legacy_key, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("VM_HOST=127.0.0.1\n")

    assert Settings(_env_file=env_file).host == "127.0.0.1"


def test_get_settings_exits_with_a_one_line_message(no_legacy_key, monkeypatch, capsys):
    monkeypatch.setenv(authkit.LEGACY_KEY_ENV, "old-shared-key")
    monkeypatch.setattr(cfg, "_settings", None)

    with pytest.raises(SystemExit) as exc:
        cfg.get_settings()

    assert exc.value.code == 1
    err = capsys.readouterr().err.strip()
    _assert_names_key_and_hint(err)
