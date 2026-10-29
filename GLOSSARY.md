# Decision Swarm

A tool for assembling several decision models into one swarm that judges a single
observation, and for packaging that swarm so other programs can query it.

## The swarm

**Observation**:
Everything a swarm judges at one moment: a bundle of named views.
_Avoid_: input, frame, payload

**View**:
One named piece of an observation, such as a screenshot or a block of text, with a
declared type.
_Avoid_: channel, field, input

**Cell**:
One sensor in a swarm: a model bound to the views it reads, plus the questions it
answers.
_Avoid_: agent, node

**Swarm**:
A fixed set of cells that all observe the same observation and answer independently.
_Avoid_: ensemble, cluster, group

**Question**:
What a cell is asked about an observation, typed as a choice, a score, or a yes/no.
_Avoid_: prompt, query, task

**Answer**:
One cell's typed response to one question, carrying a confidence.
_Avoid_: prediction, output, result

## Artifacts

**Pack**:
The build artifact describing one swarm: its cells, the models it pins, and the runtime
that serves it. A pack never contains model weights.
_Avoid_: bundle, artifact, project

**Bundle**:
A pack together with the weights and cell envs it needs, packed as one file so it can be
handed to someone who has no model cache.
_Avoid_: archive, tarball, dist

**Manifest**:
A pack's description of the swarm itself: views, cells, and questions.
_Avoid_: config, spec

**Lockfile**:
A pack's record of exactly which model revisions, commits, and digests it expects, so a
run can be verified rather than guessed.
_Avoid_: requirements, pin file

**Model registry**:
The catalog of checkpoints the tool knows about: where each one lives, how to load it,
and its license.
_Avoid_: index, catalog

**Model cache**:
The machine-wide store of downloaded checkpoints, shared by every pack.
_Avoid_: store, models folder

## Adapting a model

**Base model**:
A decision model named by the user as the starting point for specialization — always a
registry base checkpoint, never a merged checkpoint.
_Avoid_: backbone, foundation model, original

**Adapter**:
A small set of trained weight deltas (LoRA) attached to one base model, plus its
training provenance; a cell attaches at most one.
_Avoid_: fine-tune, tweak, patch

**Specialization**:
The process of training an adapter so a base model gets better at one named task ("X"),
measured before and after on that task's slice.
_Avoid_: fine-tuning run, customization

**Slice**:
A named subset of an evaluation set, used to measure whether a specialization actually
improved the behavior it targeted.
_Avoid_: split, segment, category

**Decision model**:
A model that answers typed questions — choice, score, yes/no — about one observation
with calibrated confidence; the family of models this tool swarms.
_Avoid_: classifier, predictor, policy

## Consuming a swarm

**Engine**:
The program that runs a swarm: it starts the cells, loads their models, and computes
answers.
_Avoid_: backend, server, core

**Serve**:
Hosting a pack as one long-running process that answers queries repeatedly, so models
are loaded once.
_Avoid_: host, daemon, server mode

**Transport**:
The channel a consumer uses to reach a served swarm.
_Avoid_: wrapper, wrapping, binding, protocol

**Runtime**:
The optional Python package a pack ships so that Python consumers get a command and an
import for that pack.
_Avoid_: wrapper, SDK, client

**Client**:
Consumer-side code, in whatever language, that talks to a served swarm.
_Avoid_: wrapper, binding, SDK, adapter (binding sense)

**Agent**:
A client that is a language model choosing tools, rather than a program written by
hand. It discovers a pack through tool schemas and calls it a few times per task,
so discovery and small results matter more to it than latency.
_Avoid_: bot, assistant, copilot
