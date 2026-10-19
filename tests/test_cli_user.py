"""The `voicemanager user` command group: add, list, passwd and remove.

Each test runs the CLI against a fresh store file in a temp dir (VM_DB_PATH),
so no server, model or real store is touched.
"""
import authkit  # noqa: F401  (sets up the auth env before voicemanager is imported)

import click
import pytest
from click.testing import CliRunner

from voicemanager import config
from voicemanager.auth import accounts
from voicemanager.auth.store import open_store
from voicemanager.cli import main

PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def db_path(tmp_path, monkeypatch):
    """A fresh store file per test, read through freshly loaded settings."""
    path = tmp_path / "state.db"
    monkeypatch.setenv("VM_DB_PATH", str(path))
    monkeypatch.setattr(config, "_settings", None)
    return path


def run(*args, input=None):
    return CliRunner().invoke(main, ["user", *args], input=input)


def _twice(password=PASSWORD):
    return f"{password}\n{password}\n"


def _user(name):
    with open_store() as store:
        return store.get_user(name)


def _one_line_error(result):
    """The error the command printed: exactly one line starting 'Error:'."""
    lines = [line for line in result.output.splitlines() if line.startswith("Error:")]
    assert len(lines) == 1, result.output
    return lines[0]


# --- the group ----------------------------------------------------------------

def test_user_is_a_group_with_the_four_verbs():
    group = main.commands["user"]
    assert isinstance(group, click.Group)
    assert {"add", "list", "passwd", "remove"} <= set(group.commands)


def test_password_is_never_an_option_value():
    group = main.commands["user"]
    for command in group.commands.values():
        for param in command.params:
            if isinstance(param, click.Option):
                assert param.is_flag, f"{command.name} {param.opts} takes a value"


# --- add ----------------------------------------------------------------------

def test_add_prompts_twice_and_stores_a_hash(db_path):
    result = run("add", "ann", input=_twice())

    assert result.exit_code == 0, result.output
    assert "Added user ann" in result.output
    user = _user("ann")
    assert user is not None
    assert user.password_hash and PASSWORD not in user.password_hash
    assert accounts.verify_password(PASSWORD, user.password_hash)
    assert db_path.exists()


def test_add_password_is_not_echoed():
    result = run("add", "ann", input=_twice())

    assert result.exit_code == 0, result.output
    assert PASSWORD not in result.output


def test_add_stores_name_lowercase():
    result = run("add", "Ann", input=_twice())

    assert result.exit_code == 0, result.output
    assert _user("ann") is not None


def test_add_password_from_stdin_is_non_interactive():
    result = run("add", "ann", "--password-stdin", input=f"{PASSWORD}\n")

    assert result.exit_code == 0, result.output
    assert "Password" not in result.output
    assert PASSWORD not in result.output
    assert accounts.verify_password(PASSWORD, _user("ann").password_hash)


def test_add_empty_password_from_stdin_is_refused():
    result = run("add", "ann", "--password-stdin", input="\n")

    assert result.exit_code != 0
    assert "empty" in _one_line_error(result)
    assert _user("ann") is None


def test_add_without_password_makes_a_token_only_user():
    result = run("add", "bot", "--no-password")

    assert result.exit_code == 0, result.output
    assert "Password" not in result.output
    assert _user("bot").password_hash is None


def test_add_refuses_both_password_flags():
    result = run("add", "ann", "--no-password", "--password-stdin", input=f"{PASSWORD}\n")

    assert result.exit_code != 0
    _one_line_error(result)
    assert _user("ann") is None


def test_add_mismatched_passwords_exits_non_zero_and_writes_nothing():
    result = run("add", "ann", input=f"{PASSWORD}\nsomething else\n")

    assert result.exit_code != 0
    assert "do not match" in _one_line_error(result)
    assert _user("ann") is None


def test_add_empty_password_is_refused():
    result = run("add", "ann", input="\n")

    assert result.exit_code != 0
    assert "empty" in _one_line_error(result)
    assert _user("ann") is None


def test_add_invalid_name_refused_before_any_prompt():
    result = run("add", "bad name!", input=_twice())

    assert result.exit_code != 0
    assert "Bad user name" in _one_line_error(result)
    assert "Password" not in result.output


def test_add_duplicate_refused_before_any_prompt():
    assert run("add", "ann", input=_twice()).exit_code == 0

    result = run("add", "ANN", input=_twice())

    assert result.exit_code != 0
    assert "already exists" in _one_line_error(result)
    assert "Password" not in result.output


def test_add_refuses_a_password_argument():
    result = run("add", "ann", "--password", PASSWORD)

    assert result.exit_code != 0
    assert _user("ann") is None


# --- list ---------------------------------------------------------------------

def test_list_with_no_users():
    result = run("list")

    assert result.exit_code == 0, result.output
    assert "No users" in result.output


def test_list_shows_each_user_with_password_and_token_count():
    run("add", "ann", input=_twice())
    run("add", "bot", "--no-password")
    with open_store() as store:
        accounts.mint_token(store, "bot", "console")

    result = run("list")

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    ann = next(line for line in lines if line.startswith("ann"))
    bot = next(line for line in lines if line.startswith("bot"))
    assert "password set" in ann and "0 tokens" in ann
    assert "no password" in bot and "1 token" in bot


# --- passwd -------------------------------------------------------------------

def test_passwd_sets_password_and_ends_sessions():
    run("add", "ann", input=_twice())
    with open_store() as store:
        secret = accounts.open_session(store, "ann")

    result = run("passwd", "ann", input=_twice("staple orbit lantern"))

    assert result.exit_code == 0, result.output
    assert "staple orbit lantern" not in result.output
    user = _user("ann")
    assert accounts.verify_password("staple orbit lantern", user.password_hash)
    assert not accounts.verify_password(PASSWORD, user.password_hash)
    with open_store() as store:
        assert store.get_session(accounts.hash_secret(secret)) is None


def test_passwd_from_stdin():
    run("add", "ann", input=_twice())

    result = run("passwd", "ann", "--password-stdin", input="staple orbit lantern\n")

    assert result.exit_code == 0, result.output
    assert accounts.verify_password("staple orbit lantern", _user("ann").password_hash)


def test_passwd_clear_removes_the_password():
    run("add", "ann", input=_twice())

    result = run("passwd", "ann", "--clear")

    assert result.exit_code == 0, result.output
    assert "Password" not in result.output
    assert _user("ann").password_hash is None


def test_passwd_mismatch_keeps_password_and_sessions():
    run("add", "ann", input=_twice())
    with open_store() as store:
        secret = accounts.open_session(store, "ann")

    result = run("passwd", "ann", input="one\ntwo\n")

    assert result.exit_code != 0
    assert "do not match" in _one_line_error(result)
    assert accounts.verify_password(PASSWORD, _user("ann").password_hash)
    with open_store() as store:
        assert store.get_session(accounts.hash_secret(secret)) is not None


def test_passwd_unknown_user_refused_before_prompt():
    result = run("passwd", "nobody", input=_twice())

    assert result.exit_code != 0
    assert "No user" in _one_line_error(result)
    assert "Password" not in result.output


# --- remove -------------------------------------------------------------------

def test_remove_ends_the_users_tokens_and_sessions():
    run("add", "ann", input=_twice())
    run("add", "bob", input=_twice())
    with open_store() as store:
        token = accounts.mint_token(store, "ann", "laptop")
        session = accounts.open_session(store, "ann")
        other = accounts.mint_token(store, "bob", "laptop")
        assert accounts.token_for(store, token) is not None

    result = run("remove", "Ann")

    assert result.exit_code == 0, result.output
    assert "Removed user ann" in result.output
    assert _user("ann") is None
    with open_store() as store:
        assert accounts.token_for(store, token) is None
        assert store.list_tokens("ann") == []
        assert store.get_session(accounts.hash_secret(session)) is None
        # Another user's token is untouched.
        assert accounts.token_for(store, other) is not None


def test_remove_unknown_user_exits_non_zero():
    result = run("remove", "nobody")

    assert result.exit_code != 0
    assert "No user" in _one_line_error(result)
