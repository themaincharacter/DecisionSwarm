# Adapting a decision model (for agents)

`decision-swarm adapt` turns "make a decision model better at X" into a trained LoRA
**Adapter** for a **Base model** the user names. The Base model comes from the Model
registry and is never a merged checkpoint, so there is no Adapter-on-Adapter. Training is
**pack-free** and happens before any Cell or decision exists; cells attach an Adapter
later, at most one each, and several cells may share one. Question schemas are **never
enforced**: the `questions.json` an Adapter ships is optional default question material
for authoring a Cell, not a contract.

The playbook runs end to end — build the rows, train, gate, register, install. The first
stage needs no GPU, no model download, and no model stack.

## The row schema

JSONL, one JSON object per line. Blank lines and lines starting with `#` are ignored.

| Field | Required | What it holds |
|---|---|---|
| `state` | yes | The **Observation** as text, or a JSON object of named views (the trainer serializes the object to JSON). Missing, `null`, or whitespace-only skips the row as `empty_text`. |
| `questions` | yes | Mapping of question id → Laya question object. Absent means no questions, so no items — not an error. Not a mapping skips the row as `invalid_questions`. |
| `gold` / `expected` | per question | Targets, keyed by question id. A question with neither is simply unlabelled: no training item, no skip. |

A question is trainable when it has `type` in `choice`/`score`/`noul`, non-blank
`instructions`, and the right `criteria`:

| `type` | `criteria` | Target keys |
|---|---|---|
| `choice` | object (key → description) or list, **at least 2** | criterion keys |
| `score` | list of level labels or object of levels, **at least 2** | level index text `"0"`, `"1"`, … |
| `noul` | **none** | `"false"` / `"true"` |

Anything else is an `invalid_question` skip for that question alone; the rest of the row
still counts.

### `gold` — teacher distributions

`{"probabilities": {"<label>": 0.9, ...}}` under the question id. Prefer this form: a
distribution carries confidence and the trainer's soft loss uses all of it.

- Missing keys default to 0 — except `noul`, where both sides default to 0.5.
- A distribution summing to 0 becomes uniform; the report labels it `(uniform)`.
- Negative or non-finite values, a missing `probabilities` map, or a non-mapping target
  are an `invalid_target` skip. `{}` is legal and means uniform.
- Keys matching no option are dropped by the trainer. `adapt validate` counts them and
  warns — that warning is almost always a typo (`billing` vs `billng`), and the dropped
  mass silently bends the target.

### `expected` — one plain answer

| `type` | Accepted forms |
|---|---|
| `choice` | a criterion key, raw or as its text |
| `noul` | `true`/`false`, `0`/`1`, or the text `true`/`false`/`yes`/`no`/`1`/`0` (trimmed, case-insensitive) |
| `score` | a level index text or a number in `[0, K-1]`; with object criteria, also the criterion key |

A blank string is an `empty_label` skip; anything unparseable is `invalid_target`.
`expected` is easier to write and throws the confidence away — use it when the label is
genuinely crisp, `gold` when it is a judgement.

When a question carries both, **`gold` wins**.

Extra fields beside the schema (`id`, `tags`, `language`, …) are ignored by training and
kept so you can slice the file later.

### CSV — one question per row

Two columns, names configurable. The distinct labels, in order of appearance, become the
criteria of one `choice` question:

```
decision-swarm adapt validate --data data.csv \
  --text-column body --label-column dept --question-id department \
  --instructions "Which department owns this ticket?"
```

At least two distinct labels are required, and a state with more than one view needs
JSONL.

## Validating: `decision-swarm adapt validate`

```
decision-swarm adapt validate --data train.jsonl [--eval eval.jsonl] \
  [--base <registry-id>] [--json] [--envs-dir DIR]
```

Read-only and model-free: the gate before any Specialization run. What it reports:

- **rows read / items** — items are labelled (state, question) pairs, so one row with
  three labelled questions counts three.
- **skipped** — row reasons `invalid_json`, `invalid_row`, `empty_text`,
  `invalid_questions`; question reasons `invalid_question`, `empty_label`,
  `invalid_target`. These are the names `laya.train.items_from_rows` uses, so a skip here
  is a skip there.
- **per-question balance** — item counts and effective label counts (for `gold`, the
  distribution's argmax).
- **leakage** — with `--eval`, the eval items whose (state, question) signature also
  occurs in train.
- **warnings** — class collapse (one label holds ≥90% of a question's items), questions
  declared but never labelled, dropped `gold` keys, duplicate (state, question) items in a
  single file.
- **optimizer budget** — a pure estimate: `min(400, 10%)` of items held out for
  calibration, effective batch `8 × 8`, 4 epochs. Add `--base` and, when that base is
  cached under a training package and its env is built, it runs `laya.train.dry_run` for
  exact counts; otherwise the line states why the probe was skipped. Nothing downloads.

Exit codes: `0` when the training data yields usable items, `1` when it yields none or a
file cannot be read. Warnings never change the exit code — read them anyway.

## Dataset recipes (proven upstream)

1. **Reverse-generate states from chosen answers.** Pick the answer first, then write the
   state that justifies it. The label cannot be wrong, because it defined the state.
2. **Add negatives that kill shortcuts.** If a surface feature always implies one label (a
   history prefix that always ends in `DONE`), generate counter-examples that break the
   correlation until the model must read the state rather than a marker.
3. **Keep the split clean.** Hold eval out by (state, question) signature: the same input
   must never appear in both files. Leakage and duplicates are two lines of `adapt
   validate` — read both.
4. **Aim for ~1–3k items.** Below a few hundred, training collapses to the class prior and
   the collapse warning is the tell. The default budget is 4 epochs.
5. **Phrase questions the way the Cell will ask them.** Training renders the question
   instructions into what the model reads, so a reworded question trains a different
   behaviour.

## Training: `decision-swarm adapt train`

Training runs inside a **per-base training env** the command builds for you — the base's
`package_source` plus `peft` — so training packages never reach a cell worker, and it needs a
Python install, not the frozen GUI.

| Flag | Default | Effect |
|---|---|---|
| `--base ID` | required | registry base model (custom/`load_vlm` refused), fetched into the cache if missing |
| `--data FILE` / `--eval FILE` | data required, eval none | training rows / held-out rows; no `--eval` means the calibration slice is scored in-sample |
| `--out DIR` / `--envs-dir DIR` | required / envs root | artifact directory / where the training env is built |
| `--mode lora\|head-only\|full` | `lora` | what to train |
| `--epochs N` / `--grad-accum N` | `4` / `8` | effective batch is micro-batch × grad-accum |
| `--micro-batch N` | sized to VRAM | `8` at ≥20 GB, `4` at ≥12 GB, else `1`; CPU/MPS always `1` |
| `--loss rlcd\|soft-ce` / `--seed N` / `--device auto\|cuda\|mps\|cpu` | `rlcd` / `0` / `auto` | |
| `--rank N` / `--lora-alpha N` / `--lora-dropout P` | `8` / `16` / `0.05` | LoRA shape, `lora` mode only |
| `--metric-map TEXT` | none | the X → metric mapping; recorded, never enforced |
| `--provenance KEY=VALUE` | none, repeatable | generator, prompt, seed, date |
| `--text-column` / `--label-column` / `--question-id` / `--instructions` | `text` / `label` / `label` / none | mapping for CSV input |

What lands in `--out`:

```
out/
├── adapter/       adapter.safetensors + adapter_config.json (+ questions.json)  (lora only)
├── merged/        a full checkpoint laya.load serves unchanged
├── questions.json first-seen schema per question id — optional Cell defaults
└── train_report.json
```

`lora` wraps the encoder and writes both artifacts; `head-only` freezes the encoder and `full`
trains everything. Both non-LoRA modes write only `merged/`, so neither can be registered —
register needs `adapter_config.json` and `adapter.safetensors`. An 8 GB card at micro-batch 1
takes roughly 30–60 minutes for 1–3k items × 4 epochs; bigger cards grow the micro-batch.

`eval_mode` in the report decides what the before/after deltas mean:

| `eval_mode` | When | Deltas |
|---|---|---|
| `held_out` | `--eval` given, no (state, question) signature shared with train | independent — the only mode worth gating on |
| `overlapping_eval` | `--eval` given, some items overlap training | inflated; the note says how many |
| `in_sample_calibration` | no `--eval` | fit quality, not evaluation |

The report also pins the base (repo, revision, weights sha256), counts items and skip reasons,
and records `comparison` (`delta_accuracy`, `delta_loss`, `delta_mean_confidence`), calibration
and abstention fits, per-epoch loss, training flags, `provenance`, `metric_map`, and `warnings`.

**Collapse.** If one label dominates the calibration records the fit warns, and the text lands in
`report.warnings` — class-prior collapse, what `adapt validate` flags at ≥90%. More epochs does
not fix it: add balanced rows for the thin labels (recipes 1–2). Do not ship past the warning.

## The gate: `decision-swarm adapt evaluate`

Three accuracy gates, applied before anything is registered:

| Gate | Default threshold | Fails when |
|---|---|---|
| `target` | `--min-delta 0.05` | the `--slice` slice improves by less, or has no eval items |
| `no-regression` | `--max-regression 0.02` | any other slice drops by more |
| `must-not-regress` | `--max-regression 0.02` | the guard set drops by more — blocks registration even when X improved |

**Slices are the `tags` on your rows.** Every row counts in `all`; each tag adds a slice, and a
`--slice` no row carries is empty — which fails the target gate ("the X slice must be tagged").

**Baseline**: `--base-dir DIR` (a checkpoint), else `--baseline FILE` (a prior evaluate report),
else the base's cached checkpoint; none of the three errors instead of guessing.

**The guard set** is the shipped `data/adapt/must-not-regress.jsonl` — 30 rows tagged `guard`
over five questions. Override with `--guard FILE` or `DECISION_SWARM_GUARD` only if you must.

The report (`decision-swarm-evaluate-report/1`, default `<merged>/../evaluate_report.json`)
holds the slice, both thresholds, `metric_map`, `slices` (`items`, `accuracy`,
`baseline_accuracy`), `guard`, one `{name, passed, detail}` per gate, and `pass`. Exit `0` passes,
`1` fails, `2` means the run broke. **`pass` is the single field `adapt register` reads.**

## Registering: `decision-swarm adapt register`

```
decision-swarm adapt register --adapter out/adapter --report out/evaluate_report.json \
  [--name SLUG] [--registry FILE] [--patch MANIFEST --cell ID] [--json]
```

It refuses unless the report's `pass` is `true` ("the evaluate gate has not passed (or its
report is missing); refusing to register") and both adapter files exist with a
`decision_swarm.base.registry_id`. **Ids are versioned, never overwritten:** `<base>-<name>`
(or `<base>-adapter`), then `-2`, `-3`, …; `--name` must be a slug (`a-z0-9_-`). The entry:

```json
{"id": "laya-text-refunds", "kind": "adapter", "base": "laya-text", "modality": "text",
 "path": "/abs/out/adapter", "loader": "load", "package_source": "...", "license": "...",
 "merged": {"hf_repo": "local/laya-text-refunds", "revision": "<digest>"},
 "notes": "specialization: <metric_map or unspecified>; trained <date>; evaluate gate passed"}
```

The digest is content-derived (base pin + sha256 of the adapter weights), so the printed cell
snippet is portable:

```json
{"id": "<cell-id>", "model": "laya-text", "modality": "text", "views": ["<view>"],
 "adapter": "local/laya-text-refunds@<digest>", "default_questions": {"<qid>": {"...": "..."}}}
```

`default_questions` appears only when the Adapter shipped `questions.json`, and question schemas
are **never enforced** — starting material to edit, not a contract. `cell.adapter` accepts that
`local/<name>@<digest>` ref or the registered id. `--patch MANIFEST --cell ID` is the only thing
that rewrites a Manifest, swapping that one cell's `model` and `adapter` and nothing else.

## Materialization: `decision-swarm adapt merge` and install

`decision-swarm install` builds one env per package source, then for each adapter cell:
resolve the base and merged entry, skip if cached, find the registered adapter files, require
the base in the cache (else "run `decision-swarm fetch`"), and merge inside the training env —
load base, wrap with the adapter's PEFT config, load weights, merge, save — printing one note
per cell, including why a cell was skipped.

Merged checkpoints live in the cache as `local/<name>/<digest>/`, i.e.
`~/.decision-swarm/models/local/<name>/<digest>/` (honours `--models-dir` /
`DECISION_SWARM_MODELS`) — the ordinary snapshot layout, so cache, export and bundle treat it
like any checkpoint. `fetch` materializes too, but only when the training env already exists.

Serving needs **zero loader changes**: `laya.load` reads the merged checkpoint like any other
and never sees an Adapter. `adapt merge --base ID --adapter DIR [--out DIR]` runs it standalone.

## The complete drill

"Make my decision model better at X" — a fresh agent follows exactly this.

1. **Collect rows.** Write `data/<x>/train.jsonl` and `data/<x>/eval.jsonl` per the schema, ~1–3k
   items, eval held out by (state, question) signature. Tag X rows and the must-not-hurt ones.
2. **Validate.** `decision-swarm adapt validate --data <train> --eval <eval>`; exit `1` means no
   usable items, and collapse/leakage/dropped-`gold`-key warnings are blockers.
3. **Train.**
   ```
   decision-swarm adapt train --base <base-id> \
     --data data/<x>/train.jsonl --eval data/<x>/eval.jsonl --out out/<x> \
     --metric-map "X = <question>.<metric> on the <x> slice" \
     --provenance generator=<who or what> --provenance seed=<seed>
   ```
   Then read `out/<x>/train_report.json`: `eval_mode` must be `held_out` and `warnings` must
   be free of collapse.
4. **Gate.** `decision-swarm adapt evaluate --base <base-id> --merged out/<x>/merged --data
   data/<x>/eval.jsonl --slice <x> --metric-map "..."`. Exit `0` passes, `1` fails.
5. **Retry at most twice, then stop.** On a fail, add balanced rows first; if data is not the
   problem, raise `--epochs`. After two retries, hand over the report with the failing gate
   quoted verbatim. Never lower `--min-delta` or `--max-regression` to make a run pass.
6. **Register.** `decision-swarm adapt register --adapter out/<x>/adapter --report
   out/<x>/evaluate_report.json --name <x>`, then take the printed snippet.
7. **Paste it into `swarm.json`.** Replace `<cell-id>` and `<view>`; keep or drop the questions.
8. **Install and run.** `decision-swarm install --pack .` materializes the merged checkpoint;
   `decision-swarm run --input observation.json --pretty` answers from it.

`--metric-map` records your reading of "X", including which metric on which slice means better.
That is a claim, not a fact — **the human may veto the mapping** ("better at refunds" might mean
tone, not `refund_requested`); a veto changes the question, tags and rows, so return to step 1.
