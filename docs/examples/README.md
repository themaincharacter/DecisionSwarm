# doom-swarm ViZDoom player

A driver for the `doom-swarm` pack. It keeps one `decision-swarm serve`
process warm, sends all three manifest views on every query, holds the chosen
button for 6 tics (or 35 with `--slowmo`), writes an MP4, and writes one JSONL
line per decision.

**Agents: read `HANDOFF.md` first.** It records the environment state, the four
vision-env patches, verified artifacts, calibration results, known issues, and
the next experiment.

The pack itself is not modified. This directory sits beside it:

```text
packs\
  doom-swarm\                 # the pack (do not edit)
  doom-player\                # this driver
    doom_swarm_player.py
    setup.bat
    README.md
    HANDOFF.md
    comparison-report.md        # method comparison table + fire-ok/gate results
    comparison-report.json      # machine-readable version
    multi-monster-report.md     # defend_the_center multi-enemy result
    multi_monster_test.py       # multi-monster harness
    center-adapter-report.md    # CENTER v2 adapter + hybrid3 result
    hybrid3_test.py             # CENTER/hybrid3 harness
    adapt/steer/                # steering dataset, adapter, gate report (v1)
    adapt/steer3/               # CENTER dataset, adapter, gate report (v2)
    calibration.json
    calibration-report.json
    runs\<timestamp>\
      episode.mp4
      decisions.jsonl
      summary.json
      serve_stderr.log
      frames\screen_a.png
      frames\screen_b.png
```

## Install

```bat
cd C:\Users\thema\Downloads\Decision.Swarm_win_0.5.0_x64-portable\packs\doom-player
setup.bat
```

`setup.bat` runs `doom_swarm_player.py setup`, which:

1. Installs host `vizdoom`, `pillow`, `imageio` and `imageio-ffmpeg`.
2. Runs `decision-swarm install` for the pack. The pack has two package
   sources, so this builds **two** envs.
3. Installs CUDA 12.4 `torch` + `torchvision` into **every** env and stops if
   `torch.cuda.is_available()` is `False`.
4. Applies four idempotent compatibility patches to the vision env:
   - transformers 5.20 returns `get_image_features(...).pooler_output` as a
     list; `laya-vision` 0.1.7 expects a tensor.
   - wraps the fixed Doom question in `laya.static_step.StaticStep`, capturing
     one decision as a CUDA graph. Unsupported shapes/questions fall back to
     eager inference.
   - `load_vlm` defaults to the checkpoint's native `image_size=512`. Do not
     reduce this to 384/448 for speed: those sizes lose left/right
     discrimination (the model picks `MOVE_LEFT` for a monster on the right).
   - `load_vlm` defaults to `preprocess="gpu"`, which runs the 320x240 source
     through the GPU preprocessing path. With native 512 px vision the cell is
     ~50 ms and a full query is ~80-90 ms on this RTX 2070.
5. Runs `check` without loading any model.

The CUDA wheels are several GB and the first install can take a while. If the
unpinned wheels fail to resolve, run:

```bat
python doom_swarm_player.py setup --pin-torch
```

## Check

No model is loaded:

```bat
python doom_swarm_player.py check
```

It verifies the engine, pack views/cells, cached models, host packages, the
`basic.cfg`/`basic.wad` scenario, and CUDA torch in both envs.

Manual equivalent of the torch fix:

```bat
for /D %%E in ("%USERPROFILE%\.decision-swarm\envs\*") do (
  "%%E\Scripts\python.exe" -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
  "%%E\Scripts\python.exe" -c "import torch, torchvision, sys; ok=torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, ok); sys.exit(0 if ok else 1)"
)
```

Inside a `.bat` file the loop variable is `%%E`; at an interactive prompt it is
`%E`.

## Play

Default: one decision every 6 Doom tics, about 6 decisions per second, and an
MP4 at Doom's 35 fps clock.

```bat
python doom_swarm_player.py play
```

Slower inspection mode: one decision every 35 tics. The 350 ms action hold
overshoots badly with the 512 px policy, so use this for inspecting the model,
not for score.

```bat
python doom_swarm_player.py play --slowmo
```

Useful options:

```text
--out DIR              run directory (default: runs\<timestamp>)
--seed 17              ViZDoom seed (default 17; seed 18 is a fast kill)
--tics-per-decision N  default 6, ignored by --slowmo
--no-video             skip the MP4 (JSONL is still written)
--skip-check           skip the preflight (not recommended)
--allow-network        do not pass --offline to serve
--pack-vision-prompt   use the pack's default lookout wording instead of the native prompt
--debug                echo serve stderr and tracebacks
```

With native 512 px vision the model now centres correctly and kills in a few
decisions, so recorded episodes are short. Default seed is 17; seed 18 is
similar. Use `live --episodes 0` to watch continuously. `--slowmo` still exists
for inspection, but the 35-tic hold overshoots at 512 px; it is not the scoring
configuration.

Checked-in runs:

| run | seed | tics/decision | decisions | score | warm mean | video |
|---|---:|---:|---:|---:|---:|---:|
| `runs/final-episode` | 17 | 6 | 4 | 1 | 87.3 ms | 24 frames / 0.69 s |
| `runs/final-fast` | 18 | 6 | 6 | 1 | 88.8 ms | 36 frames / 1.03 s |
| `runs/live-demo` | 18 | 6 | 37 | 1 | 90.6 ms | no MP4 (35 tps live) |
| `runs/fireok-final` | 17 | 6 | 4 | 1 | 80.3 ms | 24 frames / 0.69 s (fire-ok veto, calibrated threshold) |
| `runs/vision-final` | 17 | 6 | 4 | 1 | 83.5 ms | 24 frames / 0.69 s (vision-only baseline) |
| `runs/hybrid-final` | 17 | 6 | 4 | 1 | 91.5 ms | 24 frames / 0.69 s (vision fires + text steers) |
| `runs/adapter-smoke` | 17 | 6 | 4 | 1 | 80.1 ms | 24 frames / 0.69 s (vision fires + LoRA adapter steers, 0.6.0) |
| `runs/center-play-smoke` | 17 | 6 | 6 | 1 | 91.6 ms | 24 frames (hybrid3: vision ATTACK + CENTER text) |
| `runs/fireok-veto-050` | 17 | 6 | 50 | 0 | 80.1 ms | no MP4 (fire-ok veto at uncalibrated 0.5: negative result) |

`play` issues one explicit warmup query before decision 1 so the StaticStep CUDA
graph and the text model are already hot. That query is not written to the
JSONL; `summary.json` records it as `warmup_ms`. Query ids in the JSONL still
start at 1.

`play` returns a nonzero exit code if `KILLCOUNT` is zero, but still leaves the
artifacts for inspection.

## Live window

Watch the swarm play in a real ViZDoom window instead of making a file:

```bat
python doom_swarm_player.py live
```

It opens a 640x480 visible window, renders at the true 35 tic/s Doom clock by
default, and runs until you press Ctrl+C or close the window. On gameover it
starts a new episode. The
model always sees a 320x240 downscale of the 640x480 frame, then the checkpoint's
native 512 px processor input, so the verified policy input is unchanged. Model
inference runs in a background thread; the game does not freeze while the model
is thinking.

Useful live options:

```text
--episodes 1      stop after one gameover (default 0 = until Ctrl+C)
--ticrate 35      default; true Doom clock, GPU-preprocessed 512 px model
--ticrate 15      slower window/action cadence if you want to inspect it
--headless        no window, for testing
--resolution      320x240 or 640x480 window (model source is always 320x240,
                  then the native 512 px processor)
--seed 17         default
--out DIR         live.jsonl / live-summary.json / frames
```

The first query is a separate ~1.0 s warmup for the CUDA graph; after that the
model latency is ~90 ms at 512 px with GPU preprocessing. `live.jsonl` logs
`query_tic`, `applied_tic`, both cells, the chosen action, and latency per
decision. `live-summary.json` records episodes, scores, warmup and latency. No
MP4 is written in live mode; use `play` for that.

```bat
python doom_swarm_player.py live --episodes 1
python doom_swarm_player.py live --ticrate 15
python doom_swarm_player.py live --episodes 1 --headless
```

`runs\live-demo` is a checked-in visible 35 tic/s role-arbitration run:
37 decisions, score 1, mean model latency 90.6 ms. It has `live.jsonl`,
`live-summary.json` and the two alternating frame files, but no MP4.

## Calibration and role arbitration

`play` and `live` now default to `--aggregator role` and auto-load
`calibration.json` beside this script. The role policy is:

- `lookout` proposes the action.
- The scene geometry is the veto: `ATTACK` is allowed only when the closest
  monster is inside `center_deg` and `attack_ready=1`.
- If vision proposes `ATTACK` while the target is off-centre, the scene supplies
  the strafe direction (`MOVE_LEFT` or `MOVE_RIGHT`).
- If vision proposes a strafe, the scene direction wins when it disagrees.
- If centred but the weapon is not ready, hold `ATTACK` rather than strafing
  away from the target.
- Both cells' full probability vectors, calibrated probabilities, geometry,
  and the override reason are logged.

`--aggregator vision` uses only `lookout` (the baseline); `--aggregator hybrid`
lets vision fire and the text cell steer (no geometry); `--aggregator pool` uses
the calibrated opinion pool; `--aggregator confidence` is the old
highest-confidence rule. `--calibration FILE` overrides the auto-load.

### Calibration fit

Collect headless episodes with probabilities, then fit temperatures, the pool
weight and the geometry gate:

```bat
python doom_swarm_player.py live --headless --episodes 40 --ticrate 35 --out runs\calib-data
python doom_swarm_player.py calibrate --from runs\calib-data\live.jsonl --out calibration.json
```

The fit is temperature scaling in log-probability space plus a linear opinion
pool and a bearing-threshold gate. It writes `calibration.json` and
`calibration-report.json`.

On 967 decisions / 40 episodes collected at 512 px, GPU preprocessing, 35 tps:

| policy | validation accuracy | train accuracy | notes |
|---|---:|---:|---|
| vision only | 78.8% | 84.6% | ECE 0.046, already well calibrated |
| analyst only | 45.3% | 45.8% | ECE 0.059; near-uniform, not complementary |
| calibrated pool | 78.8% | 84.6% | fitted analyst weight **0.00** |
| bearing gate | 76.1% | 80.4% | threshold 15°; the old table's 80.4% was **train** |
| **role arbitration** | **99.7%** | 100.0% | geometry veto + direction |

The bearing-gate validation number is the important correction: it does not
generalise, because off-centre it hands the action to the analyst, whose action
accuracy is only 45.3%. Role arbitration uses the analyst's *scene input*
instead of its action, which is why it dominates. The full comparison, including
the fire-ok critic and the gate-width A/B, is in `comparison-report.md`.

**Gameplay caveat on the 99.7%:** it is agreement with the scripted 2.5° teacher,
not evidence of better play. A direct 18-seed A/B (see below) shows vision-only
actually uses fewer decisions than role at 2.5°; role at 5.0° is the
recommended configuration.

The fitted temperatures were vision 0.85 and analyst 1.80, but neither improved
validation NLL **and** ECE, so the applied calibration is identity (T=1). The
research conclusion is that the analyst model is not useful as an action voter
on this task. The swarm works because the role policy uses the analyst's
engineered **scene input** as a geometry veto, while the vision cell proposes the
action. The analyst's own action and probabilities are still logged for study.

### Gate width and the vision baseline

`center_deg` is 2.5°, but a controlled synchronous probe measured the hit
envelope at about ±5.5° (157/160 hits over seeds 1–20 and target bearings 0–6°;
the three misses were all at 5.9°). An 18-seed paired A/B makes the consequence
concrete:

| policy | decisions | kills | decisions/kill | off-centre ATTACKs | timeouts |
|---|---:|---:|---:|---:|---:|
| vision only | **61** | 18 | 3.39 | 5 | 0 |
| role, 2.5° | 70 | 18 | 3.89 | 0 | 0 |
| role, 5.0° | **59** | 18 | **3.28** | 4 | 0 |

Paired: role 2.5° is worse than vision on 4 seeds, better on 1, equal on 13.
Role 5.0° is better on 1, equal on 17, worse on 0; it also beats 2.5° on 4 seeds
and loses on none.

So **role at 2.5° is not a gameplay improvement over vision** — the table
accuracy of 99.7% measures agreement with the scripted teacher, not play. The
2.5° veto turns away from 2.9–3.9° shots that hit. Role at 5.0° is the
recommended configuration: it matches vision on almost every seed and still
vetoes wild off-centre shots (seed 10's −10.8° miss). The checked-in default is
still 2.5° so the existing verified runs remain reproducible; use
`--center-deg 5.0` to opt in.

### Vision vs text steering

`--aggregator hybrid` uses vision for firing and the text cell for steering, with
no geometry gate:

```python
if vision_choice == "ATTACK":      chosen = "ATTACK"
elif analyst_choice in ("MOVE_LEFT", "MOVE_RIGHT"): chosen = analyst_choice
else:                              chosen = vision_choice
```

On 18 paired seeds it is a large regression:

| policy | decisions | kills | decisions/kill | off-centre ATTACKs | timeouts |
|---|---:|---:|---:|---:|---:|
| vision only | **61** | 18 | 3.39 | 5 | 0 |
| vision + text steer | 111 | 18 | 6.17 | 14 | 0 |

Text steered on 31 of 111 decisions and agreed with vision only 13 times (42%).
Vision still killed every seed, but the disagreement caused turn oscillation
(seed 11: 4 → 23 decisions, seed 8: 3 → 13). The text cell's three-action
accuracy is 45.3%, so this is expected: it is not a useful steering sensor.

**HUD stability caution:** an earlier run added `hits=0` to the HUD. That one
field made the text cell choose `ATTACK` on all 61 decisions, so text never
steered and the hybrid was exactly vision-only. The HUD is back to the original
format (no `hits=`). Do not add HUD fields without re-checking the text cell's
action distribution; it is very sensitive to the string.

### Steering LoRA adapter

A 0.6.0 LoRA adapter (`laya-typed-decisions-doom-steer`) fixes the text steerer.
It answers a two-option `MOVE_LEFT`/`MOVE_RIGHT` question and turns toward the
nearest enemy by distance; there is no `ATTACK` option, so it does not train the
fire call. The full recipe and reports are in `adapt/steer/README.md`.

| slice | adapter | base |
|---|---:|---:|
| multi-enemy steering (held out, 450) | **86.9%** | 48.9% |
| single-enemy steering (held out, 114) | **100.0%** | 59.7% |
| guard (30) | **100.0%** | 50.0% |

In the game, with `--aggregator hybrid` (vision fires, text steers):

| policy | decisions | kills | decisions/kill | text steering correct |
|---|---:|---:|---:|---:|
| vision only | 61 | 18 | 3.39 | — |
| hybrid, base text | 111 | 18 | 6.17 | 13/31 (42%) |
| **hybrid, adapter** | **61** | **18** | **3.39** | **41/41 (100%)** |

The adapter brings the hybrid exactly to vision-only on all 18 `basic` seeds.
It does **not** generalise to the real multi-monster scenario, though: on
`defend_the_center` (10 seeds, `multi_monster_test.py`) its multi-enemy steering
is 22.2–50.0% vs 59.3–81.5% for base text and 78.9% for vision; the adapter
hybrid gets 10 kills / 4 deaths vs base 25/0 and vision 29/0. The synthetic
86.9% was a coverage gap: close/large monsters, non-monster objects (`Blood`),
unseen `MarineChainsawVzd`, and a much faster turn rate. See
`multi-monster-report.md`. Use the adapter for single-enemy `basic` until a v2
is trained on real multi-monster frames.

The true-isolation variant — vision fire-only `noul` + text three-way
`MOVE_LEFT`/`MOVE_RIGHT`/`STAY` — also fails with the current checkpoints. Base
text chose `STAY` 300/300; the adapter 150/156–288/300. Vision `noul` fires
off-centre (P=0.74) and the `FIRE/HOLD` choice says HOLD when centred
(P(FIRE)=0.37). Isolated policies kill 0–3 vs 18 for the earlier hybrid. See
`isolated-role-report.md`. The earlier hybrid works because it uses each cell's
trained question.

The copied pack is `packs/doom-swarm-steer`; the original `doom-swarm` is
untouched. Run it with:

```bat
python doom_swarm_player.py play --pack ..\doom-swarm-steer ^
  --engine ..\Decision.Swarm_win_0.6.0_x64-portable\decision-swarm.exe ^
  --aggregator hybrid
```

The 0.6.0 frozen launcher cannot run `adapt train`; `adapt/engine-0.6.0-src`
holds the extracted source used with the text env's Python. See
`adapt/steer/README.md`.

### CENTER v2 + `hybrid3` (middle ground)

A second adapter uses a three-option question `MOVE_LEFT` / `MOVE_RIGHT` /
`CENTER`. The `hybrid3` policy keeps vision's native 3-action question, uses
**only its `ATTACK`** as the fire signal, and lets the text cell steer:
`CENTER` holds the aim, and when no monster is visible the policy turns right to
search. The adapter is `laya-typed-decisions-doom-center@7fc341f6853930dd`;
the pack is `packs/doom-swarm-steer3`; the full report is
`center-adapter-report.md`.

Training: 2,400 train / 600 eval (real `calib-data` + real `defend_the_center`
frames + synthetic close/large/non-monster cases), 3 epochs, micro-batch 2,
**22m34s**, held-out 74.5% (multi 63.4%, real multi 84.0%, guard 91.7%), gate
PASS.

| test | decisions | kills | deaths | multi steers | multi accuracy |
|---|---:|---:|---:|---:|---:|
| basic, 18 seeds | 61 | **18** | 0 | — | 41/41 text steers |
| multi search, 10 seeds | 466 | 19 | 6 | 23 | **95.65%** |

Context on multi: vision 29/0, v1 base hybrid 25/0, v1 adapter hybrid 10/4,
role 3/7. The v2 text steering is no longer the bottleneck; vision fires
off-target 358/372 times in `defend_the_center`. Real multi eval is only 25
off-centre frames, so the 84% real-multi number is wide. The base text cell is
OOD for the three-option question (`CENTER` 88/460 on multi).

A `CENTER = keep turning` variant was tested and rejected: the v2 drops from
19/6 to 13/5 and multi steering from 95.65% to 57.89%, because it overrides a
correct centring decision. Keep `CENTER` as no-op.

```bat
python doom_swarm_player.py play --pack ..\doom-swarm-steer3 ^
  --engine ..\Decision.Swarm_win_0.6.0_x64-portable\decision-swarm.exe ^
  --aggregator hybrid3
```

## Fire-ok critic (experimental)

The analyst cell can answer an independent binary question instead of voting on
the action:

> If the player fires now, will the shot hit the monster?

This keeps both cells in the same parallel query: `lookout.next_action` remains
the native `choice`, while `analyst.next_action` becomes a `noul` question.
Nothing in the pack is modified; the driver supplies the override per query.

```bat
python doom_swarm_player.py live --fire-ok log
python doom_swarm_player.py play --fire-ok veto --fire-ok-threshold 0.45
```

| mode | question sent | effect on the action |
|---|---|---|
| `off` (default) | none; analyst keeps its choice question | none |
| `log` | fire-ok `noul` | none; `P(hit)` is logged for study |
| `veto` | fire-ok `noul` | a centred, attack-ready shot is taken only when `P(hit) >= threshold`; otherwise the player re-aims |

The critic can only remove an `ATTACK`. It can never turn a vision strafe into a
shot. `--fire-ok` requires `--aggregator role`. The threshold comes from
`calibration.json` (`fire_ok_threshold`, default `0.5`) and can be overridden
with `--fire-ok-threshold`.

`play` applies the action immediately after the query, so its `HITCOUNT` delta
labels the exact pose the model saw. `live` applies the action one decision
after the frame the model saw, so its `hits` counter is logged for inspection
but is not used to fit the threshold.

To fit the threshold on actual shots:

```bat
python doom_swarm_player.py play --seed 17 --fire-ok log --center-deg 8 --no-video --out runs\fireok-17
python doom_swarm_player.py calibrate --from runs\fireok-17 --out calibration.json
```

A run directory (or a directory whose immediate subdirectories are run
directories) is accepted. `calibrate` labels each fired decision with
`hits_delta > 0`, picks the threshold with the best balanced accuracy on a
held-out episode split, and reports AUC, accuracy, ECE, and an offline `role`
vs `role + fire-ok` comparison against the scripted geometry expert. It also
reports an auxiliary scripted-hitbox check: a shot would hit if the weapon is
ready and the closest monster is inside `--fire-ok-hitbox-deg` (default 5.5°).
If the corpus contains only hits or only misses, no discrimination can be
fitted: the threshold is set to the edge of the observed scores and `fitted` is
`false`.

`log`/`veto` replace the analyst's action `choice`, so fire-ok logs are
fire-ok-only corpora. `calibrate` detects this and preserves the existing
action calibration (temperatures, pool weight, gate) while updating
`fire_ok_threshold`.

### Fire-ok result (24 synchronous episodes, seeds 1–24)

The role policy fired 24 shots, all at `|bearing| <= 3.9°`; **all 24 hit**, so
the veto had no miss to catch in the deployment regime. The analyst's noul
probability barely moved with the scene: across all 73 decisions `P(hit)` was
0.423–0.484 (mean 0.458), and on the 24 fired shots it was 0.454–0.472, even
though every one hit. The fitted threshold is just below the smallest
`P(hit)` on a fired shot, so the veto is inert.

| quantity | value |
|---|---|
| synchronous episodes / decisions | 24 / 73 |
| fired shots labelled from `hits_delta` | 24 |
| hits / misses | **24 / 0** |
| `P(hit)` min / mean / max (all decisions) | 0.423 / 0.458 / 0.484 |
| `P(hit)` range on the 24 fired shots | 0.454–0.472 |
| ECE of `P(hit)` on fired shots (validation) | 0.542 |
| scripted hitbox AUC / balanced accuracy (validation) | 0.595 / 0.583 |
| fitted threshold | 0.4448, `fitted: false` |
| offline role vs role+fire-ok | 1.000 vs 1.000, 0 changed |

A controlled synchronous probe over seeds 1–20 and target bearings 0–6° hit
157/160 shots; the three misses were all at `|bearing| = 5.9°`. On this
scenario the geometric envelope is therefore about ±5.5°, wider than the role
policy's 2.5° centre gate.

At the uncalibrated default threshold `0.5` the veto blocks every shot because
the text cell reports `P(hit) ≈ 0.46` on shots that hit. `runs/fireok-veto-050`
is the seed-17 demonstration: 50 decisions, zero shots fired (ammo stayed 50),
score 0, with the agent oscillating between veto and geometry re-aim.

The verified artifact `runs/fireok-final` uses the calibrated threshold:
4 decisions, score 1, 80.3 ms warm mean, 24-frame MP4, `verify` PASS. Latency is
unchanged because the noul question replaces the analyst's action question in
the same parallel query.

**Conclusion:** do not enable the veto. The role gate already fires only on
shots that hit, and the noul critic is flat and under-confident. Keep
`--fire-ok log` as an observation channel; a useful critic needs a corpus with
real misses and a question the text model answers with more separation.

## Verify

```bat
python doom_swarm_player.py verify runs\<timestamp>
```

It checks that:

- `ready` arrived exactly once and serve was not restarted;
- decision indices are monotonic `1..N`;
- every line has all three views, the frame path, both cells and valid
  confidences;
- the chosen action follows the agreement/confidence rule;
- every monster's bearing sign matches its screen side;
- `KILLCOUNT` is nonzero;
- the MP4 is readable and its frame count matches the tics held;
- warm mean query latency is under ~200 ms. The StaticStep CUDA graph is what
  makes this pass consistently on the RTX 2070.

For the manual scene spot-check:

```bat
python doom_swarm_player.py verify runs\<timestamp> --write-audit-frames
```

This extracts 12 frames from the MP4 into `audit\` and prints the matching
`scene` and monster bearing for each.

## Vision prompt

The pack's default `lookout` question says "turn left/right" and "hold for six
tics". The checkpoint was trained with `laya.games.doom_question("basic", ...)`:
"strafe left/right, shoot when centered". On the same frame the pack wording
picks `MOVE_RIGHT` at 0.06 confidence; the native wording picks `MOVE_LEFT` at
0.79. The driver therefore overrides only `lookout`'s question, keeping the
question name `next_action` so the result schema is unchanged. The analyst keeps
the pack default. Pass `--pack-vision-prompt` to use the pack wording instead.

## What the driver sends

Every query carries all three views, even though `lookout` only reads the
image:

```json
{"type":"query","id":1,"views":{
  "screen_now":"...\\frames\\screen_a.png",
  "hud":"tic=14/300 ammo2=50 health=100 kills=0 attack_ready=1",
  "scene":"Zombie bearing=-8 width=0.18 distance=410"
}}
```

The image is written to one of two alternating PNGs. Frames are not
accumulated. `state.labels` categories `Self`/`Player` are filtered, labels are
sorted by `(object_name, object_id)`, and `distance` is omitted if the scenario
has cheats disabled (`POSITION_X`, `POSITION_Y`, `ANGLE` all zero).

Action aggregation is the caller's job. The default is the role policy above:
vision proposes, geometry vetoes/directs, and `ATTACK` requires a centred,
attack-ready target. `--aggregator vision` uses only `lookout` as a baseline;
`--aggregator hybrid` lets vision decide firing and the text cell decide
steering (`MOVE_LEFT`/`MOVE_RIGHT`), with no geometry gate; `--aggregator pool`
uses the calibrated opinion pool; `--aggregator confidence` keeps the old
agreement/highest-confidence rule. Both cells are always logged, with their raw
and calibrated probabilities and the override reason. With `--fire-ok log|veto`,
the driver also overrides `analyst.next_action` with the fire-ok `noul` question;
the analyst's `choice` is then `null` and its `noul`/`noul_confidence` are logged.

## Honest note for the write-up

`lookout` is a trained image cell. `analyst` is a text cell reading `hud` and a
`scene` line that this driver computes from ViZDoom labels, bounding boxes and
world positions. `scene` is engineered geometry, not learned perception. The
demo therefore shows a trained cell and a hand-built sensor disagreeing, which
is what a pack of cells is for; it is not evidence that `analyst` "sees".

The fire-ok `noul` experiment makes the same point quantitatively. The text
cell's P(hit) stayed at 0.42–0.48 while every centred, ready shot hit (24/24),
and it gave only AUC 0.60 against a scripted hitbox label. At the uncalibrated
0.5 threshold the critic vetoed every shot and scored 0. This is a negative
result for the learned critic, and a positive result for the engineered
geometry gate.

