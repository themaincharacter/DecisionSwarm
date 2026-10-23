"""Per-package virtual environments.

Cells may need mutually incompatible Python stacks — the ``laya-vision`` fork
installs its own ``laya`` module (colliding with PyPI ``laya``) and needs
``transformers >= 5.3`` — so each unique ``package_source`` gets its own venv.
Cells sharing a package source share the env but run in separate processes.
Envs are built by ``decision-swarm install --pack ...``.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from decision_swarm import registry
from decision_swarm.manifest import SwarmManifest

ENVS_DIR_NAME = ".decision-swarm/envs"
"""Default env location under the user's home directory."""

ENVS_DIR_ENV = "DECISION_SWARM_ENVS"
"""Environment variable overriding the env root."""

BASE_PYTHON_ENV = "DECISION_SWARM_PYTHON"
"""Interpreter used to create the envs; overrides the automatic choice."""

Installer = Callable[[Path, Sequence[str]], None]
"""Installs requirements into a venv given its python executable."""


class EnvError(RuntimeError):
    """Raised when an env cannot be planned or built."""


@dataclass(frozen=True)
class CellEnv:
    """One env shared by every cell with the same ``package_source``."""

    key: str
    package_source: str
    path: Path
    python: Path
    shared_by: tuple[str, ...]


def envs_dir(override: str | Path | None = None) -> Path:
    """Env root: explicit override > ``$DECISION_SWARM_ENVS`` > ``~/.decision-swarm/envs``."""
    if override is not None:
        return Path(override)
    env = os.environ.get(ENVS_DIR_ENV)
    if env:
        return Path(env)
    return Path.home() / ENVS_DIR_NAME


def env_key(package_source: str) -> str:
    """Stable filesystem-safe directory name for a package source."""
    slug = re.sub(r"[^a-z0-9]+", "-", package_source.lower()).strip("-")[:40] or "env"
    digest = hashlib.sha256(package_source.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def _venv_python(path: Path) -> Path:
    if os.name == "nt":
        return path / "Scripts" / "python.exe"
    return path / "bin" / "python"


def cell_envs(
    swarm: SwarmManifest,
    *,
    envs_root: str | Path | None = None,
    registry_entries: Mapping[str, registry.ModelEntry] | None = None,
) -> list[CellEnv]:
    """Plan one env per unique ``package_source`` (cells sharing a source share the env)."""
    grouped: dict[str, list[str]] = {}
    for cell in swarm.cells:
        package_source, _loader = registry.resolve_runtime(cell, registry_entries)
        if not package_source:
            raise EnvError(
                f"cell {cell.id!r} has no package_source; set cell.runtime or use a registry model"
            )
        grouped.setdefault(package_source, []).append(cell.id)

    envs: list[CellEnv] = []
    for source, cell_ids in grouped.items():
        key = env_key(source)
        path = envs_dir(envs_root) / key
        envs.append(
            CellEnv(
                key=key,
                package_source=source,
                path=path,
                python=_venv_python(path),
                shared_by=tuple(cell_ids),
            )
        )
    return envs


def _frozen() -> bool:
    """True inside a frozen launcher (PyInstaller), which is not a Python interpreter."""
    return bool(getattr(sys, "frozen", False))


def base_interpreter() -> Path:
    """Interpreter used to create the per-package envs.

    ``$DECISION_SWARM_PYTHON`` wins. Otherwise the running interpreter, unless
    that is the frozen launcher — which cannot create venvs — in which case the
    first ``python3``/``python`` on ``PATH`` is used.
    """
    override = os.environ.get(BASE_PYTHON_ENV)
    if override:
        return Path(override)
    if not _frozen():
        return Path(sys.executable)
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            return Path(found)
    raise EnvError(
        "no Python interpreter available to create envs with: install Python 3 "
        f"or set {BASE_PYTHON_ENV} to a Python executable"
    )


def _create_venv(base: Path, path: Path, *, with_pip: bool) -> None:
    """Create a venv at ``path`` from the ``base`` interpreter.

    ``venv.create`` always uses ``sys.executable`` — the launcher itself when
    frozen — so the base interpreter is invoked explicitly instead.
    """
    args = [str(base), "-m", "venv"]
    if not with_pip:
        args.append("--without-pip")
    args.append(str(path))
    try:
        subprocess.run(args, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EnvError(f"creating the venv at {path} with {base} failed: {exc}") from exc


def default_installer(python: Path, requirements: Sequence[str]) -> None:
    """Install requirements with the venv's pip."""
    try:
        subprocess.run([str(python), "-m", "pip", "install", *requirements], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EnvError(
            f"pip install into {python} failed ({', '.join(requirements)}): {exc}"
        ) from exc


def ensure_venv(env: CellEnv, *, with_pip: bool = True) -> Path:
    """Create the venv if it does not exist yet; return its python executable."""
    if not env.python.exists():
        _create_venv(base_interpreter(), env.path, with_pip=with_pip)
    return env.python


def install_packages(
    env: CellEnv, requirements: Sequence[str], *, installer: Installer | None = None
) -> None:
    """Install requirements into an env (injectable installer for tests)."""
    install = installer if installer is not None else default_installer
    install(env.python, list(requirements))


def build_env(
    env: CellEnv, *, installer: Installer | None = None, with_pip: bool = True
) -> CellEnv:
    """Create the env and install its package source into it."""
    ensure_venv(env, with_pip=with_pip)
    install_packages(env, [env.package_source], installer=installer)
    return env
