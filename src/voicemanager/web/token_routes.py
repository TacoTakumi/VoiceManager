"""The tokens page: the signed-in user lists, makes and revokes their own
API tokens and sets the preset voice each one speaks with.

Only the signed-in user's tokens are ever looked at, so a name that only
another user's token carries is not found: revoking it or setting its
voice answers 404, and that token stays as it was. A new token's secret is
in the one response that answers the create post, never on a later page:
the store keeps only its hash.

The voices offered are those of the TTS engine the server runs
(``app.state.tts_engine.available_voices``), not the configured list, so a
token can only name a voice this server can speak. An engine without
presets (OpenVoice clones voices instead) gets no voice control, and any
voice posted to it is refused.

Every form carries the session field; each POST checks it before it
changes anything. Every route here is a plain ``def``, so FastAPI runs it
in the thread pool and the store work stays off the event loop.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse, Response

from voicemanager.auth import accounts
from voicemanager.auth.store import Token
from voicemanager.web.render import render
from voicemanager.web.session import WebUser, csrf_refused, current_user, open_web_store
from voicemanager.web.signin_routes import TOKENS_PATH

CREATE_TOKEN_PATH = TOKENS_PATH + "/create"
REVOKE_TOKEN_PATH = TOKENS_PATH + "/revoke"
TOKEN_VOICE_PATH = TOKENS_PATH + "/voice"

SHOWN_ONCE = (
    "Copy the secret now: this page shows it once, and the server keeps only"
    " its hash."
)
NO_TOKENS = "You have no tokens."
NAME_NEEDED = "The token needs a name."
SERVER_DEFAULT = "server default"
NO_PRESETS = (
    "The running TTS engine has no preset voices, so a token cannot name one."
)
UNKNOWN_VOICE = "The running TTS engine has no voice {voice!r}."

token_router = APIRouter(include_in_schema=False)


def engine_voices(request: Request) -> list[str] | None:
    """The preset voices of the running TTS engine, or None when it has none
    (OpenVoice) or no engine is loaded."""
    engine = getattr(request.app.state, "tts_engine", None)
    voices = getattr(engine, "available_voices", None)
    return list(voices) if voices else None


def engine_default_voice(request: Request) -> str:
    """The preset the running engine speaks with when a token names none, or ''."""
    engine = getattr(request.app.state, "tts_engine", None)
    return getattr(engine, "default_voice", "") or ""


def voice_refusal(voice: str, voices: list[str] | None) -> str | None:
    """Why ``voice`` (already stripped; empty is the server default) cannot be
    stored, or None when it can."""
    if voices is None:
        return NO_PRESETS if voice else None
    if voice and voice not in voices:
        return UNKNOWN_VOICE.format(voice=voice)
    return None


def token_row(token: Token) -> dict:
    """One token as the list shows it; never the secret or its hash."""
    return {
        "name": token.name,
        "last4": token.last4 or "-",
        "voice": token.voice,
        "created": token.created,
        "last_used": token.last_used or "never",
    }


def tokens_page(request: Request, user: WebUser, *,
                minted: tuple[str, str] | None = None,
                status_code: int = 200, **context) -> Response:
    """The tokens page of ``user``. ``minted`` is a new token's (name,
    secret), shown this once; the response is never cached (render marks
    every signed-in page so). ``context`` adds a ``create_error`` or a
    ``list_error`` line."""
    with open_web_store(request) as store:
        tokens = [token_row(t) for t in accounts.list_tokens(store, user.name)]
    return render(
        request, "tokens.html", title="Tokens", status_code=status_code,
        tokens=tokens, minted=minted, voices=engine_voices(request),
        default_voice=engine_default_voice(request),
        server_default=SERVER_DEFAULT, shown_once=SHOWN_ONCE, no_tokens=NO_TOKENS,
        create_path=CREATE_TOKEN_PATH, revoke_path=REVOKE_TOKEN_PATH,
        voice_path=TOKEN_VOICE_PATH, **context)


@token_router.get(TOKENS_PATH)
def show_tokens_page(request: Request, user: WebUser = Depends(current_user)):
    return tokens_page(request, user)


@token_router.post(CREATE_TOKEN_PATH)
def create_token(request: Request, user: WebUser = Depends(current_user),
                 session: str | None = Form(default=None),
                 name: str = Form(default=""),
                 voice: str = Form(default="")):
    """Make a token for the signed-in user and show its secret, once.

    The page answers the post itself rather than redirecting, so the secret
    is in this one response and no later page has it. A blank name or a
    voice the running engine does not offer is a 400 page; a name the user
    already has is a 409 page.
    """
    refused = csrf_refused(request, session)
    if refused is not None:
        return refused
    name = name.strip()
    if not name:
        return tokens_page(request, user, status_code=400, create_error=NAME_NEEDED)
    voice = voice.strip()
    refusal = voice_refusal(voice, engine_voices(request))
    if refusal is not None:
        return tokens_page(request, user, status_code=400, create_error=refusal)
    try:
        with open_web_store(request) as store:
            secret = accounts.mint_token(store, user.name, name, voice=voice or None)
    except accounts.AccountError as exc:
        return tokens_page(request, user, status_code=409, create_error=str(exc))
    return tokens_page(request, user, minted=(name, secret))


@token_router.post(REVOKE_TOKEN_PATH)
def revoke_token(request: Request, user: WebUser = Depends(current_user),
                 session: str | None = Form(default=None),
                 name: str = Form(default="")):
    """Revoke the signed-in user's token by name, then back to the list. A
    name the user has no token of is a 404 page."""
    refused = csrf_refused(request, session)
    if refused is not None:
        return refused
    try:
        with open_web_store(request) as store:
            accounts.revoke_token(store, user.name, name)
    except accounts.AccountError as exc:
        return tokens_page(request, user, status_code=404, list_error=str(exc))
    return RedirectResponse(TOKENS_PATH, status_code=303)


@token_router.post(TOKEN_VOICE_PATH)
def set_token_voice(request: Request, user: WebUser = Depends(current_user),
                    session: str | None = Form(default=None),
                    name: str = Form(default=""),
                    voice: str = Form(default="")):
    """Set or clear the voice of the signed-in user's token, then back to the
    list. An empty voice (FastAPI reads an empty form field as a missing
    one) is the server default.

    Under an engine without presets every post is refused, as is a voice
    the running engine does not offer: a 400 page. A name the user has no
    token of is a 404 page.
    """
    refused = csrf_refused(request, session)
    if refused is not None:
        return refused
    voices = engine_voices(request)
    if voices is None:
        return tokens_page(request, user, status_code=400, list_error=NO_PRESETS)
    voice = voice.strip()
    refusal = voice_refusal(voice, voices)
    if refusal is not None:
        return tokens_page(request, user, status_code=400, list_error=refusal)
    try:
        with open_web_store(request) as store:
            accounts.set_token_voice(store, user.name, name, voice or None)
    except accounts.AccountError as exc:
        return tokens_page(request, user, status_code=404, list_error=str(exc))
    return RedirectResponse(TOKENS_PATH, status_code=303)
