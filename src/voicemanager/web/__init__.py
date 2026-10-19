"""The web UI: sign-in, sign-out and the signed-in pages, served by the API's
own app and port.

Pages are Jinja2 templates rendered server-side and sent with a strict
Content-Security-Policy; they run no script, so every flow is a plain form
post. A signed-in browser holds a server-side session in a cookie (see
:mod:`voicemanager.web.session`).
"""
