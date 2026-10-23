"""Pack builder — what the GUI's Submit emits.

A swarm pack directory contains ``swarm.json`` (manifest), ``models.lock.json``
(repo + revision + sha256 per unique checkpoint), a generated ``runtime/``
package and a README with usage for the pack. Templates for the generated
runtime live in ``templates/runtime/``.

Phase 1 lands the builder here.
"""

from __future__ import annotations

PACK_MANIFEST_NAME = "swarm.json"
"""Manifest filename inside a pack."""

PACK_LOCKFILE_NAME = "models.lock.json"
"""Model lockfile name inside a pack."""

PACK_RUNTIME_DIR = "runtime"
"""Generated Python package directory inside a pack."""
