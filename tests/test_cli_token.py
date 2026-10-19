"""The `voicemanager user token` command group: add, list and revoke.

Each test runs the CLI against a fresh store file in a temp dir (VM_DB_PATH),
so no server, model or real store is touched.
"""
import authkit  # noqa: F401  (sets up the auth env before voicemanager is imported)

import re

import click
import pytest
from click.testing import CliRunner

from voicemanager import config
from voicemanager.auth import accounts
from voicemanager.auth.store import open_store
from voicemanager.cli import main

SECRET_PATTERN = re.compile(r"vmp_[A-Za-z0-9_-]+")


@pytest.fixture(autouse=True)
def db_path(tmp_path, monkeypatch):
    """A fresh store file per test, read through freshly loaded settings."""
    path = tmp_path / "state.db"
    monkeypatch.setenv("VM_DB_PATH", str(path))
    monkeypatch.setattr(config, "_settings", None)
    return path


@pytest.fixture
def ann():
    with open_store() as store:
        accounts.add_user(store, "ann", None)
    return "ann"


def run(*args):
    return CliRunner().invoke(main, ["user", "token", *args])


def _secrets_in(text):
    return SECRET_PATTERN.findall(text)


def _add(*args):
    result = run("add", *args)
    assert result.exit_code == 0, result.output
    found = _secrets_in(result.output)
    assert len(found) == 1, result.output
    return result, found[0]


def _one_line_error(result):
    """The error the command printed: exactly one line starting 'Error:'."""
    lines = [line for line in result.output.splitlines() if line.startswith("Error:")]
    assert len(lines) == 1, result.output
    return lines[0]


# --- the group ----------------------------------------------------------------

def test_token_is_a_group_under_user_with_three_verbs():
    group = main.commands["user"].commands["token"]
    assert isinstance(group, click.Group)
    assert {"add", "list", "revoke"} <= set(group.commands)


# --- add ----------------------------------------------------------------------

def test_add_prints_the_secret_once_and_stores_only_its_hash(ann):
    result, secret = _add("ann", "console")

    assert re.fullmatch(r"vmp_[A-Za-z0-9_-]{43}", secret)
    assert result.output.count(secret) == 1
    # stdout is the secret alone, so a script can capture it.
    assert result.stdout.strip() == secret
    assert "will not be shown again" in result.stderr
    with open_store() as store:
        [token] = store.list_tokens("ann")
    assert token.name == "console"
    assert token.hash == accounts.hash_secret(secret)
    assert token.last4 == secret[-4:]
    assert token.voice == ""
    assert token.last_used is None


def test_added_secret_authenticates(ann):
    _result, secret = _add("ann", "console")

    with open_store() as store:
        token = accounts.token_for(store, secret)
    assert token is not None
    assert (token.owner, token.name) == ("ann", "console")


def test_add_accepts_owner_in_any_case(ann):
    _add("ANN", "console")

    with open_store() as store:
        assert [t.owner for t in store.list_tokens("ann")] == ["ann"]


def test_duplicate_name_for_same_user_refused(ann):
    _add("ann", "console")

    result = run("add", "ann", "console")

    assert result.exit_code != 0
    assert "already has a token named 'console'" in _one_line_error(result)
    assert not _secrets_in(result.output)
    with open_store() as store:
        assert len(store.list_tokens("ann")) == 1


def test_same_name_for_another_user_allowed(ann):
    with open_store() as store:
        accounts.add_user(store, "bob", None)

    _add("ann", "console")
    _add("bob", "console")


def test_add_for_unknown_user_refused():
    result = run("add", "nobody", "console")

    assert result.exit_code != 0
    assert "No user named 'nobody'" in _one_line_error(result)
    assert not _secrets_in(result.output)


def test_add_with_empty_name_refused(ann):
    result = run("add", "ann", "")

    assert result.exit_code != 0
    _one_line_error(result)
    assert not _secrets_in(result.output)
    with open_store() as store:
        assert store.list_tokens("ann") == []


# --- list ---------------------------------------------------------------------

def test_list_shows_details_but_never_the_secret(ann):
    _r, first = _add("ann", "console")
    with open_store() as store:
        second = accounts.mint_token(store, "ann", "laptop", voice="Carter")
        accounts.token_for(store, second)

    result = run("list", "ann")

    assert result.exit_code == 0, result.output
    assert first not in result.output and second not in result.output
    assert first[len(accounts.TOKEN_PREFIX):] not in result.output
    assert f"...{first[-4:]}" in result.output
    assert f"...{second[-4:]}" in result.output
    # Tokens are listed by name, each a header line and its indented details.
    lines = result.output.splitlines()
    laptop = next(i for i, line in enumerate(lines) if line.startswith("laptop"))
    assert lines[0].startswith("console")
    console_block = "\n".join(lines[:laptop])
    laptop_block = "\n".join(lines[laptop:])
    # A token without a voice shows a blank voice; one with a voice names it.
    assert re.search(r"^\s+voice:\s*$", console_block, re.M), console_block
    assert re.search(r"^\s+voice: Carter$", laptop_block, re.M), laptop_block
    assert "created: " in console_block and "created: " in laptop_block
    assert "last used: never" in console_block
    assert re.search(r"last used: \d{4}-\d\d-\d\dT", laptop_block), laptop_block


def test_list_with_no_tokens(ann):
    result = run("list", "ann")

    assert result.exit_code == 0, result.output
    assert "no tokens" in result.output


def test_list_unknown_user_refused():
    result = run("list", "nobody")

    assert result.exit_code != 0
    assert "No user named 'nobody'" in _one_line_error(result)


# --- revoke -------------------------------------------------------------------

def test_revoke_stops_the_secret_authenticating(ann):
    _r, secret = _add("ann", "console")
    _r, kept = _add("ann", "laptop")

    result = run("revoke", "ANN", "console")

    assert result.exit_code == 0, result.output
    assert "Revoked" in result.output and "console" in result.output
    with open_store() as store:
        assert accounts.token_for(store, secret) is None
        assert accounts.token_for(store, kept) is not None
        assert [t.name for t in store.list_tokens("ann")] == ["laptop"]


def test_revoke_unknown_token_refused(ann):
    result = run("revoke", "ann", "nothing")

    assert result.exit_code != 0
    assert "no token named 'nothing'" in _one_line_error(result)


def test_revoke_unknown_user_refused():
    result = run("revoke", "nobody", "console")

    assert result.exit_code != 0
    assert "No user named 'nobody'" in _one_line_error(result)
