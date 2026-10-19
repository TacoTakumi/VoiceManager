"""Tests for the web UI theme: the accent colour and the dark default.

Model-free and server-free: the sign-in page is fetched from create_app()
without its lifespan, against an empty store under pytest's tmp_path.
"""
import authkit  # noqa: F401  (sets up the auth env before voicemanager is imported)

import re

import pytest
from fastapi.testclient import TestClient

import voicemanager.config as config
from voicemanager.app import create_app


@pytest.fixture
def page(tmp_path, monkeypatch) -> str:
    """The sign-in page HTML, with the stylesheet inlined."""
    monkeypatch.setenv("VM_DB_PATH", str(tmp_path / "state.db"))
    monkeypatch.setattr(config, "_settings", None)
    client = TestClient(create_app(), follow_redirects=False)
    response = client.get("/signin")
    assert response.status_code == 200
    return response.text


def test_accent_is_lcars_amber(page):
    assert re.search(r"--accent:\s*light-dark\(#b45309, #f5a524\);", page)


def test_root_defaults_to_dark(page):
    root = re.search(r":root \{\s*color-scheme: ([a-z ]+);", page)
    assert root is not None
    assert root.group(1) == "dark"


def test_light_values_are_kept_for_a_theme_switch(page):
    # Every colour token still carries its light side.
    assert re.search(r"--bg:\s*light-dark\(#ffffff, #17191c\);", page)
