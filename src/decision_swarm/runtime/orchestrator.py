"""Swarm orchestrator.

Spawns one worker process per cell (2-8, in parallel), fans out each query to
every worker, and collects the raw per-cell answers. v1 performs no aggregation
or consensus — results are returned per cell; the worker topology leaves room
for neighbor links later.

Phase 2 lands spawn/collect here.
"""
