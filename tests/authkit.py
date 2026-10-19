"""Shared test auth helper - the one place the automated suite learns how to authenticate.

Test files import this module BEFORE importing anything from voicemanager:

    import authkit  # noqa: F401  (sets up the auth store at import time)
    from voicemanager.app import create_app

conftest.py imports it too, so the setup runs once at session start for every
test file. Importing it points VM_DB_PATH at a fresh auth store in a temporary
directory and seeds it with one user (USER) and one bearer token (TOKEN_NAME,
secret TOKEN). That must happen before the first get_settings() call, because
voicemanager reads its settings once and caches them.

Call sites use only these helpers and never spell out credentials themselves:

    with authkit.ws_connect(client, "/ws/stream") as ws: ...
    client.get("/info", headers=authkit.auth_headers(), params=authkit.auth_params())

The server authenticates with 'Authorization: Bearer <secret>' on the REST
endpoints and on the WebSocket handshake. auth_params() is {} and stays only so
older call sites keep working.

Rejection tests use bearer_headers() with another secret, mint()/revoke() for
extra tokens, and the LEGACY_* constants for the removed shared key, so no
collected test file has to spell the old names.
"""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

# The removed shared-key setting and its query parameter. Tests that check it is
# gone take the names from here.
LEGACY_KEY_ENV = "VM_API_KEY"
LEGACY_KEY_FIELD = "api_key"
LEGACY_KEY_PARAM = "api_key"

# The server refuses to load settings while the removed key is set, so a stale
# shell export must not leak into the suite.
os.environ.pop(LEGACY_KEY_ENV, None)

# Must run at import time, before voicemanager reads its settings (cached).
_STORE_DIR = Path(tempfile.mkdtemp(prefix="vm-test-auth-"))
atexit.register(shutil.rmtree, _STORE_DIR, ignore_errors=True)
STORE_PATH = _STORE_DIR / "state.db"
os.environ["VM_DB_PATH"] = str(STORE_PATH)

from voicemanager.auth import accounts  # noqa: E402  (after the env is set)
from voicemanager.auth.store import open_store  # noqa: E402

USER = "tester"
TOKEN_NAME = "suite"

with open_store(STORE_PATH) as _store:
    accounts.add_user(_store, USER, None)
    TOKEN = accounts.mint_token(_store, USER, TOKEN_NAME)

# Well-formed but never minted: the store has no token with this secret.
UNKNOWN_TOKEN = accounts.TOKEN_PREFIX + "x" * 43


def bearer_headers(secret: str) -> dict[str, str]:
    """An Authorization header carrying ``secret`` as a bearer token."""
    return {"Authorization": f"Bearer {secret}"}


def auth_headers() -> dict[str, str]:
    """HTTP headers that authenticate a REST call or a WebSocket handshake."""
    return bearer_headers(TOKEN)


def auth_params() -> dict[str, str]:
    """Query parameters that authenticate a REST call: none, auth is a header."""
    return {}


def legacy_query(secret: str = TOKEN) -> dict[str, str]:
    """``secret`` as the removed shared-key query parameter (the server ignores it)."""
    return {LEGACY_KEY_PARAM: secret}


def mint(name: str, *, voice: str | None = None) -> str:
    """Mint another token for USER in the suite store; its secret."""
    with open_store(STORE_PATH) as store:
        return accounts.mint_token(store, USER, name, voice=voice)


def revoke(name: str) -> None:
    """Revoke USER's token ``name`` in the suite store."""
    with open_store(STORE_PATH) as store:
        accounts.revoke_token(store, USER, name)


def ws_connect(client, path: str, **kwargs):
    """Open an authenticated WebSocket on a TestClient.

    Returns the ``client.websocket_connect(...)`` context manager, so use it as
    ``with authkit.ws_connect(client, "/ws/stream") as ws:``. Any query string on
    ``path`` is kept, headers passed in are merged over the auth header, and other
    keyword arguments pass through to websocket_connect.
    """
    kwargs["headers"] = {**auth_headers(), **kwargs.pop("headers", {})}
    return client.websocket_connect(path, **kwargs)
