"""Per-cell worker process.

Each worker loads its model once and stays warm across queries, receives only
the views its cell is bound to plus the resolved questions (query overrides win
over pack defaults), and answers them with typed ``choice`` / ``score`` /
``noul`` answers. Runs inside the env built for its cell's ``package_source``.

Phase 2 lands the worker loop here.
"""
