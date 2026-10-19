"""Browser sessions for the web UI: the session cookie, the guard every
signed-in page runs, the middleware that refreshes the cookie, and the
session field that state-changing forms echo.

A session is a random secret the browser holds in the ``vm_session`` cookie;
the auth store keeps only its SHA-256 (see :mod:`voicemanager.auth.accounts`).
Every web page except the sign-in page depends on :func:`current_user`,
which reads the cookie and resumes the session, or raises
:class:`SignInRequired`, which the app turns into a redirect to the sign-in
page (see ``signin_routes.sign_in_redirect``).

The cookie never authenticates the API (/info, /voice/upload, /ws/stream):
those read a bearer token only.

The server trusts no proxy header, so a request counts as https only when
the socket itself took it over https.
"""
from __future__ import annotations

import dataclasses
import secrets
from pathlib import Path

from fastapi import Request, Response
from fastapi.responses import PlainTextResponse
from starlette.datastructures import MutableHeaders
from starlette.requests import HTTPConnection

from voicemanager.auth import accounts
from voicemanager.auth.store import SqliteStore, open_store

SESSION_COOKIE = "vm_session"

# The request state key that holds the signed-in WebUser, once the guard has
# run, so every page rendered for the request can show who is signed in.
USER_STATE = "web_user"
# The request state key that holds a session secret whose cookie the response
# must set again, because the request moved the session's expiry.
REFRESH_STATE = "session_refresh"

# Sent with every response that carries the session secret, in a form or in a
# Set-Cookie.
SIGNED_IN_CACHE_CONTROL = "private, no-store"


class SignInRequired(Exception):
    """A web page was asked for by a browser with no live session."""


@dataclasses.dataclass(frozen=True)
class WebUser:
    """The user signed in on a request: their name, how the session was
    opened ('password') and the session's secret."""

    name: str
    method: str
    secret: str


def store_file(request: HTTPConnection) -> Path | None:
    """The store file the app was built with, or None for the configured one."""
    return getattr(request.app.state, "store_file", None)


def open_web_store(request: HTTPConnection) -> SqliteStore:
    """Open the auth store for one unit of work on ``request``.

    Call it from a plain ``def`` route or dependency (FastAPI runs those in
    the thread pool), so the SQLite work stays off the event loop.
    """
    return open_store(store_file(request))


def resume_user(request: Request) -> WebUser | None:
    """The user the request's session cookie signs in, or None.

    An expired or unknown session counts as none. When the request moves
    the session's expiry, the response refreshes the cookie (see
    :class:`SessionCookieMiddleware`). The WebUser found is kept on the
    request's state, so a second look in the same request reads no store.
    """
    known = getattr(request.state, USER_STATE, None)
    if known is not None:
        return known
    secret = request.cookies.get(SESSION_COOKIE, "")
    if not secret:
        return None
    with open_web_store(request) as store:
        session, written = accounts.resume_session(store, secret)
    if session is None:
        return None
    user = WebUser(name=session.user, method=session.method, secret=secret)
    setattr(request.state, USER_STATE, user)
    if written:
        setattr(request.state, REFRESH_STATE, secret)
    return user


def current_user(request: Request) -> WebUser:
    """The signed-in user, for a page dependency
    (``user: WebUser = Depends(current_user)``); raises SignInRequired
    without a live session. A plain def, so the store read runs in the
    thread pool."""
    user = resume_user(request)
    if user is None:
        raise SignInRequired()
    return user


def session_secret(request: Request) -> str:
    """The live session's secret: what a form that changes state echoes back
    in its hidden ``session`` field. Empty without a session."""
    user = getattr(request.state, USER_STATE, None)
    if user is not None:
        return user.secret
    return request.cookies.get(SESSION_COOKIE, "")


def request_is_https(request: HTTPConnection) -> bool:
    """Whether the browser reached the server over https, so the session
    cookie goes out Secure.

    Only the scheme the socket itself took counts: the server reads no
    forwarded header, as any client can send one.
    """
    return request.url.scheme == "https"


def set_session_cookie(response: Response, secret: str, *, secure: bool) -> None:
    """Put the session cookie for ``secret`` on ``response``, for the
    session's full life."""
    response.set_cookie(
        SESSION_COOKIE,
        secret,
        max_age=accounts.SESSION_TTL,
        httponly=True,
        samesite="lax",
        path="/",
        secure=secure,
    )


def clear_session_cookie(response: Response) -> None:
    """Tell the browser to drop its session cookie."""
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="lax")


def skip_cookie_refresh(request: Request) -> None:
    """Make the response leave the session cookie alone, as a sign-out does
    when it clears the cookie itself."""
    setattr(request.state, REFRESH_STATE, None)


class SessionCookieMiddleware:
    """Set the session cookie again on a response whose request moved the
    session's expiry.

    The guard found the session and marked the request's state; the cookie
    goes on whatever response the route made. A pure ASGI middleware, so it
    sees the state dict the route wrote to.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # The request's state is this dict; holding it here sees what the
        # route set.
        state = scope.setdefault("state", {})

        async def send_with_cookie(message) -> None:
            secret = state.get(REFRESH_STATE)
            if message["type"] == "http.response.start" and secret:
                cookie = Response()
                set_session_cookie(cookie, secret,
                                   secure=request_is_https(HTTPConnection(scope)))
                headers = MutableHeaders(scope=message)
                headers.append("set-cookie", cookie.headers["set-cookie"])
                # The cookie carries the secret, on whatever response the
                # route made: no cache may keep a copy.
                headers["cache-control"] = SIGNED_IN_CACHE_CONTROL
            await send(message)

        await self.app(scope, receive, send_with_cookie)


# --- the session field on state-changing forms ---------------------------------
#
# Every web form that changes state for a signed-in user (sign-out and the
# tokens and account pages' forms) carries the session secret in a hidden
# field named ``session`` (render passes it to every signed-in page as
# ``session``). Its POST route runs the session guard first, then, before it
# changes anything,
#
#     refused = csrf_refused(request, session)
#     if refused is not None:
#         return refused
#
# with ``session: str | None = Form(default=None)`` as a route parameter. A
# post from another site cannot read the secret, so it cannot echo it.

SESSION_FIELD = "session"
CSRF_REFUSED = "That form did not come from this session."


def csrf_refused(request: Request, field_value: str | None) -> Response | None:
    """The 403 answer for a form post whose ``session`` field is missing or
    does not match the caller's session, or None when it matches.

    Without a live session secret every post is refused. The values are
    compared in constant time as bytes, so a wrong value of any shape is a
    refusal, never a fault.
    """
    expected = session_secret(request)
    posted = (field_value or "").encode("utf-8", errors="replace")
    if expected and secrets.compare_digest(posted, expected.encode("utf-8")):
        return None
    return PlainTextResponse(CSRF_REFUSED, status_code=403)
