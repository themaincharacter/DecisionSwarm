"""``swarm.json`` schema and validation.

A pack manifest pins the shape of an observation (named views), the cells
(model + view bindings + default questions), and runtime pins for the pack.
See ``README.md`` for a full ``swarm.json`` example.

Phase 1 lands the schema types and validation here.
"""

from __future__ import annotations

SCHEMA_VERSION = 1
"""Current ``swarm.json`` schema version."""

CELL_ID_PATTERN = r"^[a-z0-9_-]+$"
"""User-named cell ids are slugs: ``a-z0-9_-``, unique within a pack."""

MIN_CELLS, MAX_CELLS = 2, 8
"""Swarm size bounds."""

QUESTION_TYPES = ("choice", "score", "noul")
"""Typed questions a cell can answer: options, rubric level, yes/no probability."""

VIEW_TYPES = ("image", "text")
"""View types an observation can carry."""
