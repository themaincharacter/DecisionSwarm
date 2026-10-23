"""Per-package virtual environments.

Cells may need mutually incompatible Python stacks — the ``laya-vision`` fork
installs its own ``laya`` module (colliding with PyPI ``laya``) and needs
``transformers >= 5.3`` — so each unique ``package_source`` gets its own venv.
Cells sharing a package source share the env but run in separate processes.
Envs are built by ``decision-swarm install --pack ...``.

Phase 2 lands the builder/installer here.
"""

from __future__ import annotations

ENVS_DIR_NAME = ".decision-swarm/envs"
"""Default env location under the user's home directory."""
