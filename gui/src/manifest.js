// Serialize UI state into the swarm.json manifest shape (plan section 4.2).
// The engine re-validates authoritatively; this module keeps the wire shape
// and the client-side sanity rules in one place.

const SLUG = /^[a-z0-9_-]+$/;
const HF_REPO = /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/;
// Portable adapter ref: local/<name>@<digest> (registry adapter ids are the
// other accepted form; `adapt register` prints this one).
const ADAPTER_REF = /^local\/[a-z0-9_-]+@[0-9a-f]{6,}$/;

function parseParams(text) {
  if (!text) return null;
  const match = /^([\d.]+)\s*([KMG])?$/i.exec(String(text).trim());
  if (!match) return null;
  const scale = { k: 1e3, m: 1e6, g: 1e9 }[(match[2] || "").toLowerCase()] || 1;
  return Math.round(parseFloat(match[1]) * scale);
}

// A cell joins the swarm (and therefore shows up in "Download") only once it
// has an id, at least one bound view, and — for custom entries — a repo id.
// Blank cells still sitting in the editor must not inflate the estimates: the
// engine only fetches models referenced by manifest cells, which always carry
// a non-empty views list.
function cellIsReady(cell, byId) {
  if (!cell.id || cell.views.length === 0) return false;
  const entry = byId.get(cell.model);
  const isCustom = !entry || entry.custom;
  return !isCustom || Boolean((cell.customRepo || "").trim());
}

// Live cluster panel metrics (plan: cells filled, unique checkpoints, params,
// est. disk, est. RAM/VRAM, workers). RAM estimate: on-disk weights x 1.3
// plus 256 MB per worker process.
export function clusterMetrics(state, models) {
  const byId = new Map(models.map((model) => [model.id, model]));
  const ready = state.cells.filter((cell) => cellIsReady(cell, byId));
  const filled = ready.length;

  const checkpoints = new Map();
  for (const cell of ready) {
    const entry = byId.get(cell.model);
    const isCustom = !entry || entry.custom;
    const hfRepo = isCustom ? cell.customRepo.trim() : entry.hf_repo;
    const revision = isCustom ? "main" : entry.revision;
    const subfolder = isCustom ? null : entry.subfolder;
    const key = `${hfRepo}@${revision}/${subfolder || ""}`;
    if (!checkpoints.has(key)) {
      checkpoints.set(key, {
        params: isCustom ? null : parseParams(entry.params),
        bytes: isCustom ? null : entry.download_bytes,
      });
    }
  }

  let params = 0;
  let bytes = 0;
  let unknown = 0;
  for (const checkpoint of checkpoints.values()) {
    if (checkpoint.params == null || checkpoint.bytes == null) {
      unknown += 1;
      continue;
    }
    params += checkpoint.params;
    bytes += checkpoint.bytes;
  }

  const workers = ready.length;
  return {
    filled,
    totalSlots: 8,
    checkpoints: checkpoints.size,
    params,
    bytes,
    ramBytes: bytes > 0 ? Math.round(bytes * 1.3 + workers * 256 * 1024 * 1024) : 0,
    workers,
    unknown,
  };
}

export function questionToManifest(question) {
  const out = { type: question.type, instructions: question.instructions };
  if (question.type === "choice") {
    out.criteria = Object.fromEntries(
      question.options.map((option) => [option.key, option.description])
    );
  } else if (question.type === "score") {
    out.criteria = [...question.levels];
  }
  return out;
}

// The wire form of a cell's adapter: the portable ref when the registry entry
// carries one (self-contained in a bundle), else the entry id, else the typed
// custom ref. Empty string means "no adapter".
export function adapterRef(cell, models) {
  if (cell.adapter === "custom-adapter-ref") return cell.customAdapterRef.trim();
  if (!cell.adapter) return "";
  const entry = models.find((model) => model.id === cell.adapter);
  if (entry && entry.merged && entry.merged.hf_repo && entry.merged.revision) {
    return `${entry.merged.hf_repo}@${entry.merged.revision}`;
  }
  return cell.adapter;
}

export function toManifest(state, models) {
  const byId = new Map(models.map((model) => [model.id, model]));
  return {
    schema_version: 1,
    name: state.name.trim(),
    input_schema: {
      views: Object.fromEntries(state.views.map((view) => [view.name, { type: view.type }])),
    },
    cells: state.cells.map((cell) => {
      const entry = byId.get(cell.model);
      const isCustom = !entry || entry.custom;
      const out = {
        id: cell.id,
        model: isCustom ? cell.customRepo.trim() : cell.model,
        modality: isCustom ? cell.customModality : entry.modality,
        views: [...cell.views],
      };
      const adapter = adapterRef(cell, models);
      if (adapter) {
        out.adapter = adapter;
      }
      if (cell.questions.length > 0) {
        out.default_questions = Object.fromEntries(
          cell.questions.map((question) => [question.name, questionToManifest(question)])
        );
      }
      const packageSource = isCustom ? cell.customPackageSource.trim() : entry.package_source;
      const loader = isCustom ? cell.customLoader.trim() : entry.loader;
      if (packageSource && loader) {
        out.runtime = { package_source: packageSource, loader: loader };
      }
      return out;
    }),
  };
}

export function validateState(state, models = []) {
  const problems = [];
  if (!state.name.trim()) {
    problems.push("pack name is required");
  }

  const viewNames = new Set();
  for (const view of state.views) {
    if (!SLUG.test(view.name)) {
      problems.push(`view name "${view.name}" must be a slug (a-z0-9_-)`);
    } else if (viewNames.has(view.name)) {
      problems.push(`duplicate view "${view.name}"`);
    }
    viewNames.add(view.name);
  }
  if (viewNames.size === 0) {
    problems.push("define at least one view");
  }

  const cellIds = new Set();
  state.cells.forEach((cell, index) => {
    const label = cell.id || `cell ${index + 1}`;
    if (!SLUG.test(cell.id)) {
      problems.push(`${label}: id must be a slug (a-z0-9_-)`);
    } else if (cellIds.has(cell.id)) {
      problems.push(`${label}: duplicate cell id`);
    }
    cellIds.add(cell.id);

    if (cell.model === "custom-hf-repo" && !HF_REPO.test(cell.customRepo.trim())) {
      problems.push(`${label}: custom model needs an org/name Hugging Face repo id`);
    }

    // Adapter attach (0..1 per cell). The engine does not validate this; the
    // editor refuses pairings that can only produce a broken merged checkpoint.
    if (cell.adapter === "custom-adapter-ref") {
      if (!ADAPTER_REF.test(cell.customAdapterRef.trim())) {
        problems.push(`${label}: adapter ref must look like local/<name>@<digest>`);
      }
    } else if (cell.adapter) {
      const adapterEntry = models.find((model) => model.id === cell.adapter && model.kind === "adapter");
      if (adapterEntry && adapterEntry.base && cell.model !== "custom-hf-repo" && adapterEntry.base !== cell.model) {
        problems.push(
          `${label}: adapter "${cell.adapter}" is trained on ${adapterEntry.base}, ` +
            `but this cell's model is ${cell.model} — pick the adapter's own base model`
        );
      }
    }

    if (cell.views.length === 0) {
      problems.push(`${label}: bind at least one view`);
    }
    for (const view of cell.views) {
      if (!viewNames.has(view)) {
        problems.push(`${label}: bound view "${view}" is not defined`);
      }
    }

    const questionNames = new Set();
    for (const question of cell.questions) {
      const questionLabel = `${label}/${question.name || "?"}`;
      if (!SLUG.test(question.name)) {
        problems.push(`${questionLabel}: question name must be a slug (a-z0-9_-)`);
      } else if (questionNames.has(question.name)) {
        problems.push(`${questionLabel}: duplicate question name`);
      }
      questionNames.add(question.name);
      if (!question.instructions.trim()) {
        problems.push(`${questionLabel}: instructions are required`);
      }
      if (question.type === "choice") {
        const keys = question.options.map((option) => option.key.trim()).filter(Boolean);
        if (keys.length < 2) {
          problems.push(`${questionLabel}: choice needs at least 2 options`);
        }
        if (new Set(keys).size !== keys.length) {
          problems.push(`${questionLabel}: duplicate option key`);
        }
      }
      if (question.type === "score") {
        const levels = question.levels.filter((level) => level.trim());
        if (levels.length < 2) {
          problems.push(`${questionLabel}: score needs at least 2 rubric levels`);
        }
      }
    }
  });
  return problems;
}
