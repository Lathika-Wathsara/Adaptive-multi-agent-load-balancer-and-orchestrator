# Created: 2026-11-18
# Created by: Lathika
# Purpose: Find the API keys for the paper services (OpenAlex, Semantic Scholar).
#          A key is looked up in the environment first, then in the .env file at the
#          top of the repo (git already ignores it). Scripts call ask_for_keys() once
#          at start: for a missing key it asks in the terminal; pressing Enter means
#          "no key, use the service the normal way" and is remembered, so the question
#          is not asked again. Library code and tests never ask, they only call get_key().

import getpass
import os
import sys
from pathlib import Path

ENV_FILE = Path(os.environ.get("KEYS_FILE") or Path(__file__).resolve().parents[2] / ".env")
SERVICES = {"OPENALEX_API_KEY": "OpenAlex", "S2_API_KEY": "Semantic Scholar"}


def read_env_file(path=None):
    """The NAME=value lines of the .env file as a dict (empty if there is no file)."""
    path = ENV_FILE if path is None else path
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def get_key(name):
    """The key for this name: environment first, then the .env file, else "" (no key)."""
    return os.environ.get(name) or read_env_file().get(name, "")


def save_key(name, value):
    """Write NAME=value into the .env file (replacing an old line) and keep the file private."""
    lines = []
    if ENV_FILE.exists():
        lines = [ln for ln in ENV_FILE.read_text(encoding="utf-8").splitlines()
                 if ln.strip().removeprefix("export ").split("=")[0].strip() != name]
    lines.append(f"{name}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        ENV_FILE.chmod(0o600)
    except OSError:
        pass


def ask_for_keys(ask=getpass.getpass, interactive=None):
    """Ask in the terminal for every key that is not set and was never skipped.

    Typing is hidden. Enter alone skips: the service is then used without a key (lower
    limits) and the skip is remembered as an empty NAME= line in .env. Does nothing when
    nobody is at the terminal (scheduled runs, tests).
    """
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        return
    for name, service in SERVICES.items():
        if get_key(name) or name in read_env_file():
            continue
        value = ask(f"{service} API key ({name}): paste it, or press Enter to continue "
                    "without a key (lower rate limits) > ").strip()
        save_key(name, value)
        if value:
            os.environ[name] = value
            print(f"  {service}: key saved in {ENV_FILE}")
        else:
            print(f"  {service}: no key, using it without one. To add one later, edit {ENV_FILE}")