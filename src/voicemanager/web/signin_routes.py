"""The web UI's front door: the home redirect, the sign-in page, password
sign-in and sign-out.

The sign-in page is the one web page a browser reaches without a session. A
password sign-in opens a session and sets its cookie (see
:mod:`voicemanager.web.session`); sign-out ends the session and clears the
cookie. Every route here is a plain ``def``, so FastAPI runs it in the
thread pool and the store work and the scrypt run stay off the event loop.

The sign-in form is drawn by :func:`sign_in_form`, which every answer on the
sign-in path goes through. It carries a pre-session token twice: in a
hidden field and in a short-lived cookie only this site's sign-in page can
set. :func:`sign_in` checks the two match before any password work, so a
form on another site cannot sign a browser in to an account of its choosing.
"""
from __future__ import annotations

import secrets
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse, Response

from voicemanager.auth import accounts
from voicemanager.web.render import STATIC_DIR, render, request_target, same_site_path
from voicemanager.web.session import (
    SIGNED_IN_CACHE_CONTROL,
    SignInRequired,
    WebUser,
    clear_session_cookie,
    csrf_refused,
    current_user,
    open_web_store,
    request_is_https,
    resume_user,
    set_session_cookie,
    skip_cookie_refresh,
)

HOME_PATH = "/"
SIGNIN_PATH = "/signin"
SIGNOUT_PATH = "/signout"
TOKENS_PATH = "/tokens"
# How a session opened by password sign-in is marked in the store.
PASSWORD_METHOD = "password"

# One message for every refused sign-in: an unknown user, a user with no
# password and a wrong password read the same.
SIGNIN_FAILED = "That user name and password do not sign in."
# A sign-in post must carry the token its form was served with, and the
# same token comes back in a cookie only this site's sign-in page can set:
# a form on another site can do neither, so it cannot sign a browser in to
# an account of the attacker's choosing.
SIGNIN_COOKIE = "vm_signin"
SIGNIN_FIELD = "signin"
SIGNIN_TOKEN_TTL = 3600
SIGNIN_FORM_STALE = "That sign-in form has expired. Sign in again."
# The favicons, served from the static folder with no session.
FAVICONS = {
    "favicon.svg": "image/svg+xml",
    "favicon.ico": "image/vnd.microsoft.icon",
    "apple-touch-icon.png": "image/png",
}
# What the sign-in page says while the store holds no user.
NO_USERS = (
    "No user exists yet, so nobody can sign in. On the server host, add one"
    " with voicemanager user add NAME (it asks for the password)."
)

web_router = APIRouter(include_in_schema=False)


def signin_token_matches(cookie: str | None, field: str | None) -> bool:
    """Whether a sign-in post's field matches its sign-in cookie, compared
    in constant time; a missing or empty either side never matches."""
    if not cookie or not field:
        return False
    return secrets.compare_digest(cookie.encode("utf-8", "replace"),
                                  field.encode("utf-8", "replace"))


def sign_in_redirect(request: Request, exc: SignInRequired) -> RedirectResponse:
    """The answer to a page asked for without a session: 303 to the sign-in
    page. A GET carries the page it asked for as ``next``, so the browser
    comes back to it after signing in; ``next`` is left out when it is the
    home page or not a path on this site. Registered on the app as the
    handler for :class:`SignInRequired`."""
    target = SIGNIN_PATH
    if request.method in ("GET", "HEAD"):
        wanted = same_site_path(request_target(request))
        if wanted != HOME_PATH:
            target = f"{SIGNIN_PATH}?next={quote(wanted, safe='')}"
    return RedirectResponse(target, status_code=303)


def sign_in_form(request: Request, next: str, *, status_code: int = 200,
                 error: str | None = None,
                 hidden: dict[str, str] | None = None) -> Response:
    """The sign-in page. ``hidden`` adds hidden fields to the form, beside
    the sign-in token.

    The page never echoes the name tried: every refusal reads the same,
    byte for byte (but for the style nonce).
    """
    with open_web_store(request) as store:
        no_users = not store.list_users()
    # A browser that already holds a sign-in cookie keeps it, so a refused
    # try answers with no cookie at all.
    held = request.cookies.get(SIGNIN_COOKIE)
    token = held or secrets.token_urlsafe(32)
    response = render(request, "signin.html", title="Sign in", status_code=status_code,
                      error=error, no_users=NO_USERS if no_users else None,
                      next=same_site_path(next),
                      hidden={**(hidden or {}), SIGNIN_FIELD: token})
    if not held:
        response.set_cookie(SIGNIN_COOKIE, token, max_age=SIGNIN_TOKEN_TTL,
                            path=SIGNIN_PATH, httponly=True, samesite="strict",
                            secure=request_is_https(request))
    return response


def favicon(name: str):
    """A route that sends the favicon ``name`` from the static folder."""
    def serve():
        return Response((STATIC_DIR / name).read_bytes(), media_type=FAVICONS[name])
    return serve


for _name in FAVICONS:
    web_router.add_api_route(f"/{_name}", favicon(_name), methods=["GET"])


@web_router.get(HOME_PATH)
def home(request: Request):
    """The tokens page for a signed-in browser, the sign-in page otherwise."""
    if resume_user(request) is None:
        return RedirectResponse(SIGNIN_PATH, status_code=303)
    return RedirectResponse(TOKENS_PATH, status_code=303)


@web_router.get(SIGNIN_PATH)
def sign_in_page(request: Request, next: str = HOME_PATH):
    # A live session has nothing to sign in here.
    if resume_user(request) is None:
        return sign_in_form(request, next)
    return RedirectResponse(same_site_path(next), status_code=303)


@web_router.post(SIGNIN_PATH)
def sign_in(request: Request, name: str = Form(default=""),
            password: str = Form(default=""),
            next: str = Form(default=HOME_PATH),
            signin: str = Form(default="")):
    # Checked before any password work: a refused post costs no scrypt run
    # and records no failure.
    if not signin_token_matches(request.cookies.get(SIGNIN_COOKIE), signin):
        return sign_in_form(request, next, status_code=403, error=SIGNIN_FORM_STALE)
    with open_web_store(request) as store:
        user, locked = accounts.check_password(store, name.strip(), password)
        if user is not None:
            secret = accounts.open_session(store, user.name, PASSWORD_METHOD)
    if locked:
        return sign_in_form(request, next, status_code=429, error=accounts.SIGNIN_LOCKED)
    if user is None:
        return sign_in_form(request, next, status_code=400, error=SIGNIN_FAILED)
    response = RedirectResponse(same_site_path(next), status_code=303)
    set_session_cookie(response, secret, secure=request_is_https(request))
    # The response carries the secret: no cache may keep it.
    response.headers["Cache-Control"] = SIGNED_IN_CACHE_CONTROL
    return response


@web_router.post(SIGNOUT_PATH)
def sign_out(request: Request, user: WebUser = Depends(current_user),
             session: str | None = Form(default=None)):
    # The form echoes the session secret in its `session` field; a post
    # without it ends nothing.
    refused = csrf_refused(request, session)
    if refused is not None:
        return refused
    with open_web_store(request) as store:
        accounts.end_session(store, user.secret)
    # This request may have moved the session's expiry; the response clears
    # the cookie, so it must not set it again too.
    skip_cookie_refresh(request)
    response = RedirectResponse(SIGNIN_PATH, status_code=303)
    clear_session_cookie(response)
    return response
