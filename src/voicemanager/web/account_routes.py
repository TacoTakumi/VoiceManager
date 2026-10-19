"""The account page: the signed-in user sees their name and changes their
password.

A password change needs the current password and the new one typed twice.
The session that makes the change stays signed in; every other session of
the user ends, so a browser left signed in elsewhere has to sign in again
with the new password. Every route here is a plain ``def``, so FastAPI runs
it in the thread pool and the store work and the scrypt run stay off the
event loop.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import Response

from voicemanager.auth import accounts
from voicemanager.web.render import render
from voicemanager.web.session import WebUser, csrf_refused, current_user, open_web_store

ACCOUNT_PATH = "/account"
CHANGE_PASSWORD_PATH = "/account/password"

PASSWORDS_DIFFER = "The two new passwords do not match."
PASSWORD_CHANGED = (
    "Your password is changed. This session stays signed in; your other"
    " sessions have ended."
)

account_router = APIRouter(include_in_schema=False)


def account_page(request: Request, user: WebUser, *, status_code: int = 200,
                 **context) -> Response:
    """The account page of ``user``; ``context`` adds the password form's
    ``password_error`` or ``password_changed`` line."""
    return render(request, "account.html", title="Account", status_code=status_code,
                  change_password_path=CHANGE_PASSWORD_PATH, **context)


@account_router.get(ACCOUNT_PATH)
def show_account_page(request: Request, user: WebUser = Depends(current_user)):
    return account_page(request, user)


@account_router.post(CHANGE_PASSWORD_PATH)
def change_password(request: Request, user: WebUser = Depends(current_user),
                    session: str | None = Form(default=None),
                    current: str = Form(default=""),
                    new: str = Form(default=""),
                    confirm: str = Form(default="")):
    """Change the signed-in user's password, keeping this session and ending
    their others.

    The session field is checked first, so a refused post changes nothing.
    A new password that is empty or not typed the same twice, and a wrong
    current password, are refused beside the form with a 400, and nothing
    changes. A wrong current password is no failed sign-in: it does not
    count toward the lockout.
    """
    refused = csrf_refused(request, session)
    if refused is not None:
        return refused
    if not new:
        return account_page(request, user, status_code=400,
                            password_error=accounts.PASSWORD_EMPTY)
    if new != confirm:
        return account_page(request, user, status_code=400,
                            password_error=PASSWORDS_DIFFER)
    with open_web_store(request) as store:
        refusal = accounts.change_password(
            store, user.name, current, new,
            keep_session_hash=accounts.hash_secret(user.secret))
    if refusal is not None:
        return account_page(request, user, status_code=400, password_error=refusal)
    return account_page(request, user, password_changed=PASSWORD_CHANGED)
