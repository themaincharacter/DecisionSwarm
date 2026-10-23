// Serialize UI state into the swarm.json manifest shape (plan section 4.2).
// The engine re-validates authoritatively; this module keeps the wire shape
// and the client-side sanity rules in one place.

const SLUG = /^[a-z0-9_-]+$/;
const HF_REPO = /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/;

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

export function validateState(state) {
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
