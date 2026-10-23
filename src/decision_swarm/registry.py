"""Model registry access.

``registry/models.json`` lists the known checkpoints (the Laya family, the
``thaitea/laya-vision`` checkpoints) plus a free-form entry for any Hugging
Face repo id. Entries carry the loader name, ``package_source`` for per-package
envs, size estimates and the weights license (shown to users before download).

Phase 1 lands loading and lookup here.
"""

from __future__ import annotations

REGISTRY_RELATIVE_PATH = ("registry", "models.json")
"""Location of the registry file inside the repository."""
