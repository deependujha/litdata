"""Maintain the PyTorch support policy declared in `.github/torch-support.json`.

Subcommands:
    check   validate the policy file and the ``torch`` lower bound in ``requirements.txt``
    export  publish ``latest`` / ``previous`` / ``minimum`` as step outputs for the CI matrix
    bump    refresh ``latest`` / ``previous`` from the newest stable release on PyPI
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_POLICY_FILE = _ROOT / ".github" / "torch-support.json"
_REQUIREMENTS_FILE = _ROOT / "requirements.txt"
_TORCH_REQUIREMENT = re.compile(r"^torch\s*>=\s*(?P<version>[\w.]+)", re.MULTILINE)
_MINOR = re.compile(r"^\d+\.\d+$")


def _load_policy() -> dict:
    """Return the parsed support policy."""
    return json.loads(_POLICY_FILE.read_text())


def _write_policy(policy: dict) -> None:
    """Write the support policy back, keeping the formatting prettier expects."""
    _POLICY_FILE.write_text(json.dumps(policy, indent=2) + "\n")


def _as_tuple(version: str) -> tuple:
    """Return a comparable tuple for a version string such as ``2.14`` or ``2.1.0``."""
    return tuple(int(part) for part in version.split("."))


def _declared_lower_bound() -> str:
    """Return the ``torch`` lower bound declared in ``requirements.txt``."""
    match = _TORCH_REQUIREMENT.search(_REQUIREMENTS_FILE.read_text())
    if match is None:
        raise SystemExit(f"no `torch >=...` requirement found in {_REQUIREMENTS_FILE.name}")
    return match.group("version")


def _latest_on_pypi() -> str:
    """Return the newest stable PyTorch release published on PyPI, as ``MAJOR.MINOR``."""
    with urllib.request.urlopen("https://pypi.org/pypi/torch/json", timeout=30) as response:
        version = json.load(response)["info"]["version"]
    major, minor = version.split(".")[:2]
    return f"{major}.{minor}"


def _emit(**outputs: str) -> None:
    """Send step outputs to GitHub Actions, or to stdout when running locally."""
    lines = [f"{key}={value}" for key, value in outputs.items()]
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as fo:
            fo.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def check() -> None:
    """Fail if the policy is malformed or drifted apart from ``requirements.txt``."""
    policy = _load_policy()
    errors = []

    for key in ("latest", "previous"):
        if not _MINOR.match(policy[key]):
            errors.append(f"`{key}` must be a MAJOR.MINOR version, got {policy[key]!r}")
    if errors:
        raise SystemExit("\n".join(errors))

    latest, previous, minimum = policy["latest"], policy["previous"], policy["minimum"]
    if _as_tuple(previous) >= _as_tuple(latest):
        errors.append(f"`previous` ({previous}) must be older than `latest` ({latest})")
    if _as_tuple(minimum) > _as_tuple(previous):
        errors.append(f"`minimum` ({minimum}) must not be newer than `previous` ({previous})")

    declared = _declared_lower_bound()
    if declared != minimum:
        errors.append(
            f"{_REQUIREMENTS_FILE.name} pins `torch >={declared}` but the policy `minimum` is {minimum};"
            " update whichever is wrong"
        )

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"supported PyTorch: {latest} and {previous}, minimum {minimum}")


def export() -> None:
    """Publish the supported versions so other jobs can build their matrix from them."""
    policy = _load_policy()
    _emit(latest=policy["latest"], previous=policy["previous"], minimum=policy["minimum"])


def bump() -> None:
    """Move the policy forward when PyTorch publishes a newer minor release."""
    policy = _load_policy()
    current, newest = policy["latest"], _latest_on_pypi()

    if _as_tuple(newest) <= _as_tuple(current):
        print(f"policy is up to date: latest PyTorch is {newest}, policy says {current}")
        _emit(changed="false", latest=current)
        return

    major, minor = (int(part) for part in newest.split("."))
    policy["latest"] = newest
    policy["previous"] = f"{major}.{minor - 1}" if minor else current
    _write_policy(policy)
    print(f"bumped: latest {current} -> {policy['latest']}, previous -> {policy['previous']}")
    _emit(changed="true", latest=policy["latest"], previous=policy["previous"])


def main() -> None:
    """Run the requested subcommand."""
    commands = {"check": check, "export": export, "bump": bump}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=list(commands))
    commands[parser.parse_args().command]()


if __name__ == "__main__":
    sys.exit(main())
