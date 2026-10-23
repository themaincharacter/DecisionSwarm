"""Shared model cache.

Checkpoints are downloaded once into a local cache (default
``~/.decision-swarm/models``; override with ``--models-dir`` or
``DECISION_SWARM_MODELS``) and content-addressed by repo + revision: one copy
per unique checkpoint no matter how many cells use it, verified against
``models.lock.json`` instead of re-fetched, and offline once warm. The CLI and
GUI read the same cache path.

Phase 1 lands fetch, sha256 verify and dedup here.
"""

from __future__ import annotations

MODELS_DIR_ENV = "DECISION_SWARM_MODELS"
"""Environment variable overriding the cache location."""

DEFAULT_MODELS_DIR_NAME = ".decision-swarm/models"
"""Cache location under the user's home directory by default."""
