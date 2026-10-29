// Cell grid + questions editor for the Decision Swarm GUI.
//
// Rendering is imperative and small: structural changes (add/remove/type
// switches) re-render a section; plain typing mutates state directly so focus
// is never stolen.

import { listModels, downloadModels, runningInTauri, submitPack } from "./bridge.js";
import { adapterRef, clusterMetrics, toManifest, validateState } from "./manifest.js";

const $ = (selector) => document.querySelector(selector);

let models = [];
const state = {
  name: "my-swarm",
  views: [{ name: "", type: "text" }],
  cells: [blankCell(0), blankCell(1)],
};

function modelChoices() {
  // adapters are not models: they attach to one (the cell editor offers them
  // in their own picker)
  return models.filter((model) => model.kind !== "adapter");
}

function adapterChoices() {
  return models.filter((model) => model.kind === "adapter");
}

function blankCell(index) {
  const firstModel = modelChoices()[0];
  return {
    id: `cell_${index + 1}`,
    model: firstModel ? firstModel.id : "custom-hf-repo",
    adapter: "",
    customAdapterRef: "",
    customRepo: "",
    customModality: "text",
    customPackageSource: "",
    customLoader: "load",
    views: [],
    questions: [],
  };
}

function blankQuestion(index) {
  return {
    name: `question_${index + 1}`,
    type: "noul",
    instructions: "",
    options: [
      { key: "a", description: "" },
      { key: "b", description: "" },
    ],
    levels: ["low", "high"],
  };
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null) continue;
    if (key === "className") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "value" || key === "checked" || key === "disabled") node[key] = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child != null) node.append(child);
  }
  return node;
}

function select(options, value, onChange) {
  const node = el("select", {});
  for (const [optionValue, label] of options) {
    node.append(el("option", { value: optionValue }, label));
  }
  node.value = value;
  node.addEventListener("change", (event) => onChange(event.target.value));
  return node;
}

function textInput(value, onInput, placeholder) {
  return el("input", {
    type: "text",
    value,
    placeholder,
    onInput: (event) => onInput(event.target.value),
  });
}

function button(label, onClick, className = "mini") {
  return el("button", { type: "button", className, onClick }, label);
}

function modelLabel(model) {
  return model.custom ? `${model.id} (any HF repo)` : model.id;
}

function licenseLine(entry) {
  if (!entry) return el("p", { className: "hint" }, "custom model — license unknown until you declare it");
  const nonCommercial = (entry.license || "").includes("CC BY-NC-SA");
  return el(
    "p",
    { className: nonCommercial ? "license warn" : "license" },
    `License: ${entry.license || "unknown (user-supplied)"}${nonCommercial ? " · non-commercial weights" : ""}`
  );
}

/* ---------- rendering ---------- */

function renderModelsTable() {
  const body = $("#models-table tbody");
  body.replaceChildren(
    ...models.map((model) =>
      el(
        "tr",
        { dataset: { modelId: model.id } },
        el("td", {}, model.id),
        el(
          "td",
          {},
          model.kind === "adapter"
            ? `adapter → ${model.base || "?"}`
            : model.hf_repo
              ? `${model.hf_repo}${model.subfolder ? `/${model.subfolder}` : ""}`
              : model.id
        ),
        el("td", {}, model.modality || "custom"),
        el("td", {}, model.params || "—"),
        el(
          "td",
          { className: (model.license || "").includes("CC BY-NC-SA") ? "warn" : "" },
          model.license || "unknown (user-supplied)"
        )
      )
    )
  );
  $("#models-status").textContent = `${models.length} models · engine: ${
    runningInTauri() ? "CLI sidecar" : "browser mock"
  }`;
}

function renderViews() {
  const editor = $("#views-editor");
  editor.replaceChildren(
    ...state.views.map((view, index) =>
      el(
        "div",
        { className: "view-row" },
        textInput(view.name, (value) => {
          view.name = value;
          renderCells();
          renderValidation();
        }, "view name (slug)"),
        select([["image", "image"], ["text", "text"]], view.type, (value) => {
          view.type = value;
          renderCells();
        }),
        button("remove", () => {
          state.views.splice(index, 1);
          renderAll();
        })
      )
    )
  );
}

function cellCard(cell, index) {
  const entry = models.find((model) => model.id === cell.model);
  const isCustom = !entry || entry.custom;
  return el(
    "article",
    { className: "cell-card", dataset: { cell: String(index) } },
    el(
      "header",
      { className: "cell-head" },
      el("span", { className: "cell-slot" }, `#${index + 1}`),
      textInput(cell.id, (value) => {
        cell.id = value;
        renderValidation();
      }, "cell id (slug)"),
      button("×", () => removeCell(index), "mini danger")
    ),
    el(
      "label",
      { className: "field" },
      "Model ",
      select(modelChoices().map((model) => [model.id, modelLabel(model)]), cell.model, (value) => {
        cell.model = value;
        renderCells();
        renderValidation();
      })
    ),
    licenseLine(entry),
    el(
      "label",
      { className: "field" },
      "Adapter ",
      select(
        [["", "none"]].concat(
          adapterChoices().map((adapter) => [adapter.id, `${adapter.id} → ${adapter.base || "?"}`]),
          [["custom-adapter-ref", "custom ref…"]]
        ),
        cell.adapter,
        (value) => {
          cell.adapter = value;
          renderCells();
          renderValidation();
        }
      )
    ),
    cell.adapter === "custom-adapter-ref"
      ? textInput(cell.customAdapterRef, (value) => {
          cell.customAdapterRef = value;
          renderValidation();
        }, "local/<name>@<digest>")
      : cell.adapter
        ? el(
            "p",
            { className: "hint" },
            `Runs the merged checkpoint (${adapterRef(cell, models)}) — install materializes it.`
          )
        : null,
    isCustom
      ? el(
          "div",
          { className: "custom-fields" },
          textInput(cell.customRepo, (value) => {
            cell.customRepo = value;
            renderValidation();
          }, "org/name (Hugging Face repo)"),
          select([["text", "text"], ["image+text", "image+text"]], cell.customModality, (value) => {
            cell.customModality = value;
          }),
          textInput(cell.customPackageSource, (value) => {
            cell.customPackageSource = value;
          }, "package source (pip requirement or git URL)"),
          textInput(cell.customLoader, (value) => {
            cell.customLoader = value;
          }, "loader (load | load_vlm)")
        )
      : null,
    el(
      "fieldset",
      { className: "views-bind" },
      el("legend", {}, "Views"),
      state.views.map((view, viewIndex) =>
        el(
          "label",
          { className: "check" },
          el("input", {
            type: "checkbox",
            checked: cell.views.includes(view.name),
            onChange: (event) => toggleCellView(cell, view.name, event.target.checked),
          }),
          ` ${view.name || `view_${viewIndex + 1}`} (${view.type})`
        )
      )
    ),
    questionsEditor(cell, index)
  );
}

function questionsEditor(cell, cellIndex) {
  return el(
    "section",
    { className: "questions" },
    el(
      "div",
      { className: "questions-head" },
      el("h4", {}, "Default questions"),
      button("+ question", () => {
        cell.questions.push(blankQuestion(cell.questions.length));
        renderCells();
        renderValidation();
      })
    ),
    cell.questions.length === 0
      ? el("p", { className: "hint" }, "No default questions — queries may add them.")
      : cell.questions.map((question, questionIndex) => questionRow(cell, cellIndex, question, questionIndex))
  );
}

function questionRow(cell, cellIndex, question, questionIndex) {
  return el(
    "div",
    { className: "question-row", dataset: { q: String(questionIndex) } },
    el(
      "div",
      { className: "question-head" },
      textInput(question.name, (value) => {
        question.name = value;
        renderValidation();
      }, "question name (slug)"),
      select([["noul", "noul (yes/no)"], ["choice", "choice"], ["score", "score"]], question.type, (value) => {
        question.type = value;
        renderCells();
        renderValidation();
      }),
      button("×", () => {
        cell.questions.splice(questionIndex, 1);
        renderCells();
        renderValidation();
      }, "mini danger")
    ),
    textInput(question.instructions, (value) => {
      question.instructions = value;
      renderValidation();
    }, "instructions"),
    criteriaEditor(question)
  );
}

function criteriaEditor(question) {
  if (question.type === "noul") {
    return el("p", { className: "hint" }, "No criteria — noul is a yes/no probability.");
  }
  if (question.type === "choice") {
    return el(
      "div",
      { className: "criteria" },
      question.options.map((option, optionIndex) =>
        el(
          "div",
          { className: "option-row" },
          textInput(option.key, (value) => {
            option.key = value;
            renderValidation();
          }, "key"),
          textInput(option.description, (value) => {
            option.description = value;
          }, "description"),
          button("×", () => {
            question.options.splice(optionIndex, 1);
            renderCells();
            renderValidation();
          }, "mini danger")
        )
      ),
      button("+ option", () => {
        question.options.push({ key: `opt_${question.options.length + 1}`, description: "" });
        renderCells();
        renderValidation();
      })
    );
  }
  return el(
    "div",
    { className: "criteria" },
    question.levels.map((level, levelIndex) =>
      el(
        "div",
        { className: "option-row" },
        textInput(level, (value) => {
          question.levels[levelIndex] = value;
          renderValidation();
        }, "rubric level"),
        button("×", () => {
          question.levels.splice(levelIndex, 1);
          renderCells();
          renderValidation();
        }, "mini danger")
      )
    ),
    button("+ level", () => {
      question.levels.push("");
      renderCells();
      renderValidation();
    })
  );
}

function renderCells() {
  const grid = $("#cells-grid");
  grid.replaceChildren(...state.cells.map(cellCard));
  $("#cell-count-label").textContent = `${state.cells.length} / 8`;
  $("#cell-count").value = String(state.cells.length);
}

function formatBytes(value) {
  if (!value) return "—";
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} GB`;
  return `${(value / 1e6).toFixed(0)} MB`;
}

function formatParams(value) {
  if (!value) return "—";
  return `${(value / 1e6).toFixed(0)}M`;
}

function metricTile(label, value) {
  return el("div", { className: "metric" },
    el("span", { className: "metric-label" }, label),
    el("span", { className: "metric-value" }, value));
}

function renderCluster() {
  const metrics = clusterMetrics(state, models);
  const suffix = metrics.unknown > 0 ? " ± custom" : "";
  $("#cluster-metrics").replaceChildren(
    metricTile("cells filled", `${metrics.filled} / ${metrics.totalSlots}`),
    metricTile("unique checkpoints", String(metrics.checkpoints)),
    metricTile("total params", formatParams(metrics.params) + suffix),
    metricTile("est. disk", formatBytes(metrics.bytes) + suffix),
    metricTile("est. RAM/VRAM", formatBytes(metrics.ramBytes) + suffix),
    metricTile("workers", String(metrics.workers))
  );
  $("#cluster-hint").textContent =
    "Estimates cover ready cells only (id + at least one view). disk = checkpoint weights · " +
    "RAM/VRAM ≈ weights × 1.3 + 256 MB per worker." +
    (metrics.unknown > 0 ? " Custom models carry no size data yet." : "");
  const progress = $("#download-progress");
  progress.max = Math.max(1, metrics.checkpoints);
}

function appendLog(line, className = "") {
  const log = $("#engine-log");
  const entry = document.createElement("div");
  if (className) entry.className = className;
  entry.textContent = line;
  log.append(entry);
  log.scrollTop = log.scrollHeight;
}

async function handleDownload() {
  const progress = $("#download-progress");
  progress.value = 0;
  appendLog("Downloading models to the shared cache…");
  let done = 0;
  try {
    await downloadModels(JSON.stringify(toManifest(state, models)), null, (line) => {
      done += 1;
      progress.value = Math.min(done / Math.max(1, Number(progress.max)), 1);
      appendLog(line);
    });
    appendLog("Download complete.");
  } catch (error) {
    appendLog(`Download failed: ${error}`, "error");
  }
}

async function handleSubmit() {
  const problems = validateState(state, models);
  if (problems.length > 0) {
    appendLog(`Cannot submit: ${problems[0]}`, "error");
    return;
  }
  const dest = $("#submit-dest").value.trim();
  const exportBundle = $("#submit-export").checked;
  appendLog(`Submitting pack to ${dest}…`);
  try {
    const summary = await submitPack(JSON.stringify(toManifest(state, models)), dest, null, exportBundle);
    appendLog(`Built ${summary.pack} — cells: ${(summary.cells || []).join(", ")}`);
    if (summary.bundle) {
      appendLog(`Bundled pack + weights → ${summary.bundle}`);
    } else if (exportBundle) {
      appendLog(
        "No bundle written: the models are not in the cache yet. Pack the swarm first with Download, then Submit again (the pack directory itself is complete).",
        "error"
      );
    }
  } catch (error) {
    appendLog(`Submit failed: ${error}`, "error");
  }
}

function renderValidation() {
  renderCluster();
  const problems = validateState(state, models);
  const node = $("#validation");
  if (problems.length === 0) {
    node.className = "validation ok";
    node.textContent = `Configuration valid — ${manifestSummary()}`;
  } else {
    node.className = "validation bad";
    node.textContent = problems.length === 1 ? problems[0] : `${problems.length} problems · ${problems[0]}`;
  }
}

function manifestSummary() {
  const manifest = toManifest(state, models);
  const questions = manifest.cells.reduce(
    (total, cell) => total + Object.keys(cell.default_questions || {}).length,
    0
  );
  return `${manifest.cells.length} cells · ${Object.keys(manifest.input_schema.views).length} views · ${questions} questions`;
}

function renderAll() {
  renderViews();
  renderCells();
  renderValidation();
}

/* ---------- state mutations ---------- */

function toggleCellView(cell, viewName, checked) {
  if (checked) {
    if (!cell.views.includes(viewName)) cell.views.push(viewName);
  } else {
    cell.views = cell.views.filter((name) => name !== viewName);
  }
  renderValidation();
}

function removeCell(index) {
  if (state.cells.length <= 2) return;
  state.cells.splice(index, 1);
  renderAll();
}

function setCellCount(count) {
  while (state.cells.length < count) state.cells.push(blankCell(state.cells.length));
  while (state.cells.length > count) state.cells.pop();
  renderAll();
}

/* ---------- wiring ---------- */

function boot() {
  $("#swarm-name").addEventListener("input", (event) => {
    state.name = event.target.value;
    renderValidation();
  });
  $("#cell-count").addEventListener("input", (event) => setCellCount(Number(event.target.value)));
  $("#add-view").addEventListener("click", () => {
    state.views.push({ name: "", type: "text" });
    renderAll();
  });
  $("#download-models").addEventListener("click", handleDownload);
  $("#submit-pack").addEventListener("click", handleSubmit);
  window.__decisionSwarm = {
    state,
    toManifest,
    validateState,
    clusterMetrics,
    renderAll,
    getManifest: () => toManifest(state, models),
  };
}

async function loadModels() {
  try {
    models = await listModels();
    renderModelsTable();
    for (const cell of state.cells) {
      // the sentinel is a user choice, not a default
      if (cell.model === "custom-hf-repo" || !modelChoices().some((model) => model.id === cell.model)) {
        cell.model = modelChoices()[0].id;
      }
    }
    renderAll();
  } catch (error) {
    $("#models-status").textContent = `Failed to load models: ${error}`;
  }
}

boot();
loadModels();
