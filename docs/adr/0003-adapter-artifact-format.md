# 0003. Adapter artifact format and merged-checkpoint serving

Date: 2026-10-29
Status: Accepted

## Context

A Specialization trains an Adapter against a user-named Base model, and `adapt train`
emits two artifacts: the Adapter (weight deltas — tiny, portable, pinnable) and a merged
checkpoint (a whole Decision model — fat, loadable by everything that already loads
checkpoints). The Adapter must carry enough provenance to reproduce an evaluation months
later, and the Engine must serve it without learning a new load path.

Shapes considered:

- **(a) merged checkpoint only** — nothing new to load, but each Specialization costs a
  full checkpoint copy and the deltas cannot be shared or pinned separately.
- **(b) Adapter only, merged at load time** — one Base model in the cache plus many tiny
  Adapters, via `peft` inside cell envs; needs `peft` in every serving env and changes
  to the loader.
- **(c) both, serving a merged checkpoint materialized at fetch/install** — the Adapter
  is the shipping and pinning unit; serving reuses the existing checkpoint path.

(c) keeps the loader untouched and works offline; its cost is disk.

## Decisions

1. **An Adapter is a directory.** `adapter/` holds `adapter.safetensors` (the LoRA
   weights, renamed from PEFT's `adapter_model.safetensors`), `adapter_config.json`, and
   optionally `questions.json`.

2. **`adapter_config.json` is PEFT community fields plus a `decision_swarm` block.** The
   community fields stay top level so PEFT tooling can read the file. The block records
   format `decision-swarm-adapter/1`, the pinned Base model (registry id, hf_repo,
   subfolder, revision, sha256 of its `model.safetensors`), trainable and total parameter
   counts, seeds, and pointers to `questions.json` and `../train_report.json`.

3. **`questions.json` is optional default question material.** It is never validated
   against a pack and never enforced; a Cell may start from it when it is authored.

4. **Serving materializes a merged checkpoint per (Base model, Adapter) pair** at
   `fetch`/`install`, into the model cache as `local/<name>@<content-digest>`, so cache,
   export and bundle treat it like any other checkpoint. The loader serves it unchanged.

5. **Two alternatives are rejected for v1.** Load-time Adapters (shape (b), with `peft`
   in cell envs) are deferred until disk cost or several Adapters per cell demand them;
   a first-class PEFT patch upstream would delete the loop glue, but the artifact format
   must not depend on it.

## Consequences

- Adapter files are single-digit MB and pinned twice: sha256 in `adapter_config.json`
  and in `models.lock.json` when the pack is bundled.
- Merged checkpoints are fat, one full copy per (Base model, Adapter) pair; the cache
  dedups them by digest, so repeated fetches of one pair cost nothing extra.
- Adapters are local-first: `adapt register` writes a versioned registry entry; `--push`
  (HF Hub upload) is a later opt-in flag. A Cell attaches at most one Adapter, unvalidated,
  so a wrong pairing shows up as a bad answer, not as a rejection.
