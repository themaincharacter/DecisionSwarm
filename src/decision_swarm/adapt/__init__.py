"""Agent-driven LoRA adaptation for decision models.

``decision-swarm adapt`` turns \"make a decision model better at X\" into a
trained LoRA adapter for a user-named base model, which cells then attach
(each cell its own adapter).  Training is pack-free and happens before any
cell decisions exist.  The playbook lives in ``docs/agents/adapt.md``.
"""

from __future__ import annotations

from decision_swarm import envs

TRAINING_PACKAGES = ("peft",)
"""Installed beside the base's ``package_source`` in the training env."""


class AdaptError(RuntimeError):
    """Raised when an adaptation run cannot proceed."""


def training_env(package_source: str, *, envs_root: str | None = None) -> envs.CellEnv:
    """The env ``adapt train`` runs in: the base's package source plus training extras.

    Kept separate from serving envs so training packages (``peft``) never leak
    into a cell worker's environment.
    """
    key = envs.env_key(f"{package_source} [adapt]")
    path = envs.envs_dir(envs_root) / key
    return envs.CellEnv(
        key=key,
        package_source=package_source,
        path=path,
        python=envs._venv_python(path),
        shared_by=(),
    )
