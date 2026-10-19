"""Page rendering for the web UI: the template environment, the strict
Content-Security-Policy and the helpers that keep redirects on this site.

Every page goes out through :func:`render`, which makes a fresh style nonce
per response, puts it on the one inline style block and in the policy. The
policy's ``default-src 'none'`` with no script-src means no script runs, and
the pages carry none: every flow is a plain form post.
"""
from __future__ import annotations

import secrets
from pathlib import Path
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import HTMLResponse
from jinja2 import ChoiceLoader, Environment, FileSystemLoader, select_autoescape

from voicemanager.web.session import (
    SIGNED_IN_CACHE_CONTROL,
    USER_STATE,
    session_secret,
)

WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

# The static folder is on the search path so base.html can include the
# stylesheet inline: pages load no external CSS.
templates = Environment(
    loader=ChoiceLoader([FileSystemLoader(TEMPLATES_DIR), FileSystemLoader(STATIC_DIR)]),
    autoescape=select_autoescape(["html"]),
)


def content_security_policy(nonce: str) -> str:
    """The policy a web page is sent with. Pages run no script and load
    only the favicons: the one style block carries this response's ``nonce``."""
    return (
        "default-src 'none'; img-src 'self'; "
        f"style-src 'nonce-{nonce}'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    )


def render(request: Request, template, *, status_code: int = 200, **context) -> HTMLResponse:
    """Render ``template`` (a name in the templates folder, or a Template
    made with ``templates.from_string``) as an HTML response.

    A page rendered for a signed-in request gets the user's name as ``user``
    and the session secret as ``session``, so base.html shows who is signed
    in and a sign-out form that echoes the secret; any other page (the
    sign-in page) gets neither. The page depends on the cookie, so the
    response varies on Cookie, and a signed-in page is never cached.
    """
    signed_in = getattr(request.state, USER_STATE, None)
    nonce = secrets.token_urlsafe(16)
    page = templates.get_template(template) if isinstance(template, str) else template
    html = page.render(
        csp_nonce=nonce,
        user=signed_in.name if signed_in else None,
        session=session_secret(request) if signed_in else "",
        **context,
    )
    headers = {
        "Vary": "Cookie",
        "Content-Security-Policy": content_security_policy(nonce),
    }
    if signed_in:
        # The page carries the session secret in its forms: no cache, shared
        # or the browser's own, may keep a copy.
        headers["Cache-Control"] = SIGNED_IN_CACHE_CONTROL
    return HTMLResponse(html, status_code=status_code, headers=headers)


def request_target(request: Request) -> str:
    """The requested path and query as the client sent them, still quoted."""
    raw = request.scope.get("raw_path")
    path = raw.decode("latin-1") if raw else quote(request.url.path)
    query = request.url.query
    return f"{path}?{query}" if query else path


def same_site_path(target: str | None) -> str:
    """``target`` when it is a path on this site, else /. A path starts with
    a single / and holds no backslash or control character, so the browser
    cannot read it as another host."""
    if (not target or not target.startswith("/") or target.startswith("//")
            or "\\" in target
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in target)):
        return "/"
    return target
