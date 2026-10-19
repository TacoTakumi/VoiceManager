"""Bearer-token auth for the REST endpoints and the WebSocket handshake.

A request authenticates with ``Authorization: Bearer <secret>`` and nothing
else: a token in a query parameter is ignored. The token is looked up in the
auth store on every request, so a revoke or a user removal made through the
CLI applies at the next one.

The lookup opens a SQLite store, so it is blocking work: it runs in the thread
pool through ``run_in_threadpool``, never on the event loop. The secret is never
logged.
"""
from __future__ import annotations

import logging

from fastapi import HTTPException, Request, WebSocket
from starlette.concurrency import run_in_threadpool
from starlette.requests import HTTPConnection

from voicemanager.auth import accounts
from voicemanager.auth.store import Token, open_store

logger = logging.getLogger(__name__)

BEARER_PREFIX = "Bearer "


def bearer_secret(authorization: str | None) -> str | None:
    """The secret in an Authorization header value, or None when it carries none."""
    if not authorization or not authorization.startswith(BEARER_PREFIX):
        return None
    return authorization[len(BEARER_PREFIX):].strip() or None


def lookup_token(secret: str) -> Token | None:
    """The token ``secret`` is, or None. Blocking: call it through run_in_threadpool."""
    with open_store() as store:
        return accounts.token_for(store, secret)


async def authenticate(connection: HTTPConnection) -> Token | None:
    """The token a request or WebSocket handshake carries, or None when it has no valid one."""
    secret = bearer_secret(connection.headers.get("authorization"))
    if secret is None:
        return None
    return await run_in_threadpool(lookup_token, secret)


async def require_token(request: Request) -> Token:
    """FastAPI dependency: the request's token, or 401 when it is missing or invalid."""
    secret = bearer_secret(request.headers.get("authorization"))
    if secret is None:
        raise HTTPException(
            status_code=401,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = await run_in_threadpool(lookup_token, secret)
    if token is None:
        raise HTTPException(
            status_code=401,
            detail="invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


async def websocket_token(websocket: WebSocket) -> Token | None:
    """Pre-accept check for a WebSocket: its token, or None after rejecting it.

    Call it before ``accept()``. Without a valid bearer token the socket is
    closed with 1008 while still unaccepted, so no WebSocket is ever established:
    the ASGI server answers the upgrade request with HTTP 403 instead.
    """
    token = await authenticate(websocket)
    if token is None:
        logger.info("WebSocket rejected: no valid bearer token")
        await websocket.close(code=1008)
    return token
