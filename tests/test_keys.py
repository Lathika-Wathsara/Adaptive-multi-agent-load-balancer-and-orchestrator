# Created: 2026-11-18
# Created by: Lathika
# Purpose: Tests for src/tools/keys.py: finding keys, saving them, and asking for them
#          in the terminal (with a fake input function, nobody types anything).

import pytest

from src.tools import keys


@pytest.fixture(autouse=True)
def private_env(tmp_path, monkeypatch):
    monkeypatch.setattr(keys, "ENV_FILE", tmp_path / ".env")
    for name in keys.SERVICES:
        monkeypatch.delenv(name, raising=False)


def test_missing_file_means_no_key():
    assert keys.get_key("OPENALEX_API_KEY") == ""


def test_file_formats_are_read():
    keys.ENV_FILE.write_text(
        '# my keys\nexport OPENALEX_API_KEY="abc123"\nS2_API_KEY = xyz\n\nnot a line\n')
    assert keys.get_key("OPENALEX_API_KEY") == "abc123"
    assert keys.get_key("S2_API_KEY") == "xyz"


def test_environment_beats_the_file(monkeypatch):
    keys.ENV_FILE.write_text("OPENALEX_API_KEY=from_file\n")
    monkeypatch.setenv("OPENALEX_API_KEY", "from_env")
    assert keys.get_key("OPENALEX_API_KEY") == "from_env"


def test_save_replaces_an_old_line_and_keeps_others():
    keys.ENV_FILE.write_text("OTHER=1\nOPENALEX_API_KEY=old\n")
    keys.save_key("OPENALEX_API_KEY", "new")
    assert keys.read_env_file() == {"OTHER": "1", "OPENALEX_API_KEY": "new"}
    assert (keys.ENV_FILE.stat().st_mode & 0o777) == 0o600


def test_nobody_at_the_terminal_means_no_question():
    asked = []
    keys.ask_for_keys(ask=lambda p: asked.append(p) or "x", interactive=False)
    assert asked == [] and not keys.ENV_FILE.exists()


def test_pasted_keys_are_saved_and_used(monkeypatch):
    answers = iter(["  oa-key  ", "s2-key"])
    keys.ask_for_keys(ask=lambda p: next(answers), interactive=True)
    assert keys.get_key("OPENALEX_API_KEY") == "oa-key"
    assert keys.get_key("S2_API_KEY") == "s2-key"


def test_enter_skips_and_is_remembered():
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return ""

    keys.ask_for_keys(ask=ask, interactive=True)
    assert len(asked) == 2 and keys.get_key("OPENALEX_API_KEY") == ""
    keys.ask_for_keys(ask=ask, interactive=True)          # second run: not asked again
    assert len(asked) == 2


def test_a_key_that_is_already_set_is_not_asked_for(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "have")
    asked = []
    keys.ask_for_keys(ask=lambda p: asked.append(p) or "", interactive=True)
    assert len(asked) == 1 and "Semantic Scholar" in asked[0]


def test_prompt_tells_the_user_that_enter_skips():
    asked = []
    keys.ask_for_keys(ask=lambda p: asked.append(p) or "", interactive=True)
    assert "press Enter" in asked[0]