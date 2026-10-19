"""`voicemanager`: the command-line entry point of the server.

`voicemanager serve` starts the server. `python -m voicemanager` runs the same
serve path, so both start the server the same way.

`voicemanager user ...` manages the server's users, local on the server host:
every command opens the auth store at VM_DB_PATH. The account rules live in
:mod:`voicemanager.auth.accounts`; the user commands only ask, print and map a
refusal to a non-zero exit with a one-line message.

A password is typed at a hidden prompt, or read from stdin with
``--password-stdin``; it is never an option value, so it lands neither in
shell history nor in the process list. It is never printed or logged.

`voicemanager user token ...` makes, lists and revokes a user's bearer
tokens. A token's secret is printed once, by ``add``, and never logged; the
store keeps only its hash and last 4 characters.

A token may name the preset voice it speaks with (``add --voice``, or
``voice`` later). The name is checked against the configured TTS engine's
presets, read from its manifest or voices directory without loading a model
(see :mod:`voicemanager.voices`); an engine without presets (OpenVoice)
refuses every voice.
"""
import logging

import click

from voicemanager import voices
from voicemanager.auth import accounts
from voicemanager.auth.store import open_store
from voicemanager.config import get_settings


def configure_logging() -> None:
    """Set up stdlib logging (INFO, timestamped) for every command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def run_server() -> None:
    """Start the VoiceManager server with uvicorn.

    Settings are loaded first so configuration errors (e.g., the removed VM_API_KEY still set)
    appear as clear error messages before uvicorn binds the port.
    """
    # Import here so logging is configured before any module-level loggers fire
    from voicemanager.config import get_settings
    import uvicorn

    settings = get_settings()  # Fails fast with a clear error on bad config
    uvicorn.run(
        "voicemanager.app:create_app",
        host=settings.host,
        port=settings.port,
        factory=True,
        # Do not trust X-Forwarded-* headers: the client address is the peer's.
        proxy_headers=False,
    )


@click.group()
def main() -> None:
    """VoiceManager: streaming STT and TTS server."""
    configure_logging()


@main.command("serve")
def serve() -> None:
    """Start the server on the configured host and port (VM_HOST, VM_PORT)."""
    run_server()


# --- users ---------------------------------------------------------------------

def _fail(message: str) -> None:
    raise click.ClickException(message)


def _ask_password() -> str:
    """A password typed twice at a hidden prompt, or a refusal.

    Two different entries or an empty one fail at once instead of asking
    again. The prompts go to stderr, so stdout keeps to what the command prints.
    """
    first = click.prompt("Password", hide_input=True, default="",
                         show_default=False, err=True)
    if not first:
        _fail("A password may not be empty.")
    second = click.prompt("Repeat password", hide_input=True, default="",
                          show_default=False, err=True)
    if first != second:
        _fail("The two passwords do not match; nothing was changed.")
    return first


def _read_password_stdin() -> str:
    """The first line of stdin, without its line ending, as the password."""
    password = click.get_text_stream("stdin").readline().rstrip("\r\n")
    if not password:
        _fail("A password may not be empty.")
    return password


def _get_password(from_stdin: bool) -> str:
    return _read_password_stdin() if from_stdin else _ask_password()


@main.group("user")
def user_group() -> None:
    """Manage users and their tokens (local on the server host)."""


@user_group.command("add")
@click.argument("name")
@click.option("--password-stdin", "password_stdin", is_flag=True,
              help="Read the password from the first line of stdin instead of "
                   "asking for it at the terminal.")
@click.option("--no-password", "no_password", is_flag=True,
              help="Add the user without a password: they cannot sign in to "
                   "the web pages, but can own tokens.")
def user_add(name: str, password_stdin: bool, no_password: bool) -> None:
    """Add the user NAME (1 to 64 of a-z 0-9 . _ - @, stored lowercase).

    Asks for the password twice at the terminal unless --password-stdin or
    --no-password is given.
    """
    if password_stdin and no_password:
        _fail("Give --password-stdin or --no-password, not both.")
    with open_store() as store:
        try:
            # Check before asking, so nobody types a password for a user who
            # cannot be added.
            name = accounts.check_new_user(store, name)
            password = None if no_password else _get_password(password_stdin)
            user = accounts.add_user(store, name, password)
        except accounts.AccountError as exc:
            _fail(str(exc))
    kind = "with a password" if user.password_hash else "no password, tokens only"
    click.echo(f"Added user {user.name} ({kind}).")


@user_group.command("list")
def user_list() -> None:
    """List users: whether a password is set, and how many tokens each has."""
    with open_store() as store:
        users = store.list_users()
        if not users:
            click.echo("No users yet. 'voicemanager user add <name>' adds the first.")
            return
        for user in users:
            tokens = store.list_tokens(user.name)
            password = "password set" if user.password_hash else "no password"
            count = f"{len(tokens)} token" + ("" if len(tokens) == 1 else "s")
            click.echo(f"{user.name}  {password}, {count}")


@user_group.command("passwd")
@click.argument("name")
@click.option("--password-stdin", "password_stdin", is_flag=True,
              help="Read the new password from the first line of stdin instead "
                   "of asking for it at the terminal.")
@click.option("--clear", is_flag=True,
              help="Remove the password instead of setting one: the user can "
                   "no longer sign in to the web pages, but keeps their tokens.")
def user_passwd(name: str, password_stdin: bool, clear: bool) -> None:
    """Set a new password for NAME, asked twice; end their sessions."""
    if password_stdin and clear:
        _fail("Give --password-stdin or --clear, not both.")
    with open_store() as store:
        try:
            # Check before asking, so nobody types a password for no user.
            name = accounts.require_user(store, name).name
            password = None if clear else _get_password(password_stdin)
            user = accounts.set_password(store, name, password)
        except accounts.AccountError as exc:
            _fail(str(exc))
    if user.password_hash:
        click.echo(f"Set a new password for {user.name}; their sessions have ended.")
    else:
        click.echo(f"Cleared the password of {user.name}; their sessions have ended.")


@user_group.command("remove")
@click.argument("name")
def user_remove(name: str) -> None:
    """Remove NAME with every token and session they hold."""
    with open_store() as store:
        try:
            removed = accounts.remove_user(store, name)
        except accounts.AccountError as exc:
            _fail(str(exc))
    click.echo(f"Removed user {removed} with their tokens and sessions.")


# --- tokens --------------------------------------------------------------------

def _check_voice(voice: str) -> str:
    """``voice`` if the configured TTS engine has that preset, or a refusal."""
    try:
        return voices.check_voice(get_settings(), voice)
    except voices.VoiceError as exc:
        _fail(str(exc))


@user_group.group("token")
def user_token_group() -> None:
    """Manage a user's bearer tokens: add, list, voice and revoke."""


@user_token_group.command("add")
@click.argument("user")
@click.argument("name")
@click.option("--voice", default=None, metavar="VOICE",
              help="The preset voice the token speaks with; one of the "
                   "configured TTS engine's presets. Default: the server's.")
def user_token_add(user: str, name: str, voice: str | None) -> None:
    """Make the token NAME for USER and print its secret once.

    The secret is the only line on stdout, so a script can capture it; the
    note around it goes to stderr. Only its hash is stored.
    """
    # Check the voice first, so a refused one makes no token.
    if voice is not None:
        voice = _check_voice(voice)
    with open_store() as store:
        try:
            secret = accounts.mint_token(store, user, name, voice=voice)
        except accounts.AccountError as exc:
            _fail(str(exc))
    click.echo(f"Token {name!r} for {user.lower()}. "
               "Store it now, it will not be shown again:", err=True)
    click.echo(secret)


@user_token_group.command("list")
@click.argument("user")
def user_token_list(user: str) -> None:
    """List USER's tokens with last 4, voice and times; never the secret."""
    with open_store() as store:
        try:
            tokens = accounts.list_tokens(store, user)
        except accounts.AccountError as exc:
            _fail(str(exc))
    owner = user.lower()
    if not tokens:
        click.echo(f"{owner} has no tokens. "
                   f"'voicemanager user token add {owner} <name>' makes one.")
        return
    for token in tokens:
        click.echo(f"{token.name}  ...{token.last4}")
        # An empty voice means the server default.
        click.echo(f"  voice: {token.voice}".rstrip())
        click.echo(f"  created: {token.created}")
        click.echo(f"  last used: {token.last_used or 'never'}")


@user_token_group.command("voice")
@click.argument("user")
@click.argument("name")
@click.argument("voice", required=False)
@click.option("--clear", is_flag=True,
              help="Remove the token's voice instead of setting one: it then "
                   "speaks with the server default.")
def user_token_voice(user: str, name: str, voice: str | None, clear: bool) -> None:
    """Set the preset VOICE that USER's token NAME speaks with.

    VOICE must be one of the configured TTS engine's presets; an engine
    without presets (OpenVoice) refuses every voice. --clear removes it.
    """
    if clear == (voice is not None):
        _fail("Give a VOICE or --clear, not both and not neither.")
    if voice is not None:
        voice = _check_voice(voice)
    with open_store() as store:
        try:
            voice = accounts.set_token_voice(store, user, name, voice)
        except accounts.AccountError as exc:
            _fail(str(exc))
    owner = user.lower()
    if voice:
        click.echo(f"The token {name!r} of {owner} now speaks with {voice}.")
    else:
        click.echo(f"Cleared the voice of the token {name!r} of {owner}; "
                   "it speaks with the server default.")


@user_token_group.command("revoke")
@click.argument("user")
@click.argument("name")
def user_token_revoke(user: str, name: str) -> None:
    """Delete USER's token NAME; it stops working at its next request."""
    with open_store() as store:
        try:
            accounts.revoke_token(store, user, name)
        except accounts.AccountError as exc:
            _fail(str(exc))
    click.echo(f"Revoked the token {name!r} of {user.lower()}.")
