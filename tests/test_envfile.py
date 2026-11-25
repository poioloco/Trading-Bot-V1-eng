"""Tests of the .env file loader (API keys)."""
from __future__ import annotations

import os

from src.utils.envfile import load_env


def test_load_env_handles_crlf_and_comments(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_bytes(
        b"# comment\r\n"
        b"TEST_KEY_ALPHA=abc123\r\n"
        b"TEST_KEY_EMPTY=\r\n"
        b"TEST_KEY_SPACES =  with spaces  \r\n"
        b'TEST_KEY_QUOTES="in quotes"\n'
    )
    for k in ("TEST_KEY_ALPHA", "TEST_KEY_EMPTY",
              "TEST_KEY_SPACES", "TEST_KEY_QUOTES"):
        monkeypatch.delenv(k, raising=False)

    n = load_env(env)

    assert n == 3
    assert os.environ["TEST_KEY_ALPHA"] == "abc123"             # no leftover \r
    assert "TEST_KEY_EMPTY" not in os.environ                   # empty ones ignored
    assert os.environ["TEST_KEY_SPACES"] == "with spaces"
    assert os.environ["TEST_KEY_QUOTES"] == "in quotes"


def test_load_env_does_not_overwrite(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TEST_KEY_ALPHA=new\n", encoding="utf-8")
    monkeypatch.setenv("TEST_KEY_ALPHA", "existing")

    n = load_env(env)

    assert n == 0
    assert os.environ["TEST_KEY_ALPHA"] == "existing"


def test_load_env_missing_file(tmp_path):
    assert load_env(tmp_path / "nonexistent.env") == 0
