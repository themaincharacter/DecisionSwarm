"""JSON-over-stdio protocol between the orchestrator and cell workers.

One request/response stream per worker: the orchestrator sends resolved
questions plus only the views the cell is bound to; the worker replies with the
result schema (per-cell answers, model, latency, warnings).

Phase 2 lands the message types and framing here.
"""
