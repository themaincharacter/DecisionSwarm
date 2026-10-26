#!/usr/bin/env python3
"""Stamp one version everywhere it is declared, then resync both lockfiles.

Seven places hold a version and none of them follow each other automatically:

============================================  ==================================
``pyproject.toml``                            wheel / sdist metadata
``src/decision_swarm/__init__.py``            ``decision-swarm --version``
``gui/src-tauri/tauri.conf.json``             bundle + installer names (Tauri reads this)
``gui/package.json``                          inert (``private: true``), kept in step by convention
``gui/src-tauri/Cargo.toml``                  crate metadata; Tauri overrides it when bundling
``gui/package-lock.json``                     ``npm ci`` never writes a lockfile
``gui/src-tauri/Cargo.lock``                  cargo only rewrites it when Cargo.toml changes
============================================  ==================================

The two lockfiles are the ones that go stale: ``npm ci`` is contractually
read-only, and ``cargo`` only refreshes its lock when ``Cargo.toml`` itself
changes. So ``npm install --package-lock-only`` and ``cargo metadata`` do that
part, and this script runs them for you.

    python scripts/bump_version.py 0.5.0     # stamp every source and resync locks
    python scripts/bump_version.py --check   # verify they agree; exit 1 on drift
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SIMPLE_VERSION = re.compile(r"^(?:version|__version__) = \"(?P<v>[^\"]+)\"$", re.MULTILINE)
JSON_VERSION = re.compile(r"^  \"version\": \"(?P<v>[^\"]+)\",$", re.MULTILINE)

STAMPED: dict[str, re.Pattern[str]] = {
    "pyproject.toml": SIMPLE_VERSION,
    "src/decision_swarm/__init__.py": SIMPLE_VERSION,
    "gui/package.json": JSON_VERSION,
    "gui/src-tauri/tauri.conf.json": JSON_VERSION,
    "gui/src-tauri/Cargo.toml": SIMPLE_VERSION,
}

PACKAGE_LOCK = "gui/package-lock.json"
CARGO_LOCK = "gui/src-tauri/Cargo.lock"
CARGO_LOCK_PACKAGE = "decision-swarm-gui"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _stamp(text: str, pattern: re.Pattern[str], version: str) -> str:
    def replace(match: re.Match[str]) -> str:
        start, end = match.span("v")
        whole = match.group(0)
        offset = match.start()
        return whole[: start - offset] + version + whole[end - offset :]

    return pattern.sub(replace, text)


def declared() -> dict[str, str]:
    """Every stamped file's current version, in file order."""
    return {
        relative: pattern.search(_read(relative)).group("v")  # type: ignore[union-attr]
        for relative, pattern in STAMPED.items()
    }


def lockfile_versions() -> dict[str, str]:
    """The version each lockfile records for the GUI package itself."""
    package_lock = json.loads(_read(PACKAGE_LOCK))
    cargo_lock = _read(CARGO_LOCK)
    match = re.search(
        r'\[\[package\]\]\nname = "' + re.escape(CARGO_LOCK_PACKAGE) + r'"\nversion = "([^"]+)"',
        cargo_lock,
    )
    return {
        PACKAGE_LOCK: package_lock.get("version", "<missing>"),
        f"{PACKAGE_LOCK} packages['']": package_lock.get("packages", {}).get("", {}).get(
            "version", "<missing>"
        ),
        CARGO_LOCK: match.group(1) if match else "<missing>",
    }


def write(version: str) -> None:
    for relative, pattern in STAMPED.items():
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        stamped = _stamp(text, pattern, version)
        if stamped != text:
            path.write_text(stamped, encoding="utf-8")
            print(f"stamped  {relative}")


def _run(command: list[str], cwd: Path, *, quiet: bool = False) -> bool:
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.DEVNULL if quiet else subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0 and not quiet:
        detail = (result.stderr or "").strip().splitlines()
        print(f"  {command[0]} failed: {detail[-1] if detail else 'unknown error'}", file=sys.stderr)
    return result.returncode == 0


def resync_locks() -> None:
    """Regenerate the two lockfiles' root version with their own tooling."""
    npm = shutil.which("npm")
    if npm is None:
        print("warning: npm not found; gui/package-lock.json left stale", file=sys.stderr)
    else:
        gui = ROOT / "gui"
        if not _run(
            [npm, "install", "--package-lock-only", "--offline", "--ignore-scripts"], gui
        ) and not _run([npm, "install", "--package-lock-only", "--ignore-scripts"], gui):
            print("warning: could not update gui/package-lock.json", file=sys.stderr)
        else:
            print("synced   gui/package-lock.json")

    cargo = shutil.which("cargo")
    if cargo is None:
        print("warning: cargo not found; gui/src-tauri/Cargo.lock left stale", file=sys.stderr)
        return
    src_tauri = ROOT / "gui" / "src-tauri"
    metadata = [cargo, "metadata", "--format-version", "1"]
    if _run([*metadata, "--offline"], src_tauri, quiet=True) or _run(
        metadata, src_tauri, quiet=True
    ):
        print("synced   gui/src-tauri/Cargo.lock")
    else:
        print("warning: could not update gui/src-tauri/Cargo.lock", file=sys.stderr)


def check() -> int:
    everywhere = {**declared(), **lockfile_versions()}
    distinct = sorted(set(everywhere.values()))
    if len(distinct) == 1:
        print(f"all {len(everywhere)} version sources agree on {distinct[0]}")
        return 0
    print("version sources disagree:", file=sys.stderr)
    width = max(len(name) for name in everywhere)
    for name, found in everywhere.items():
        print(f"  {name:<{width}}  {found}", file=sys.stderr)
    print("run: python scripts/bump_version.py <version>", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stamp one version across every source.")
    parser.add_argument("version", nargs="?", help="new version, e.g. 0.5.0")
    parser.add_argument(
        "--check", action="store_true", help="verify the sources agree, change nothing"
    )
    args = parser.parse_args(argv)

    if args.check:
        return check()
    if not args.version:
        parser.error("a version is required unless --check is given")
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        parser.error(f"{args.version!r} is not a plain x.y.z version")

    write(args.version)
    resync_locks()
    return check()


if __name__ == "__main__":
    sys.exit(main())
