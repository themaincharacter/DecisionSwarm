// Cell grid + questions editor for the Decision Swarm GUI.
//
// Rendering is imperative and small: structural changes (add/remove/type
// switches) re-render a section; plain typing mutates state directly so focus
// is never stolen.

import { listModels, runningInTauri } from "./bridge.js";
import { toManifest, validateState } from "./manifest.js";

const $ = (selector) => document.querySelector(selector);

let models = [];
const state = {
  name: "my-swarm",
  views: [{ name: "", type: "text" }],
  cells: [blankCell(0), blankCell(1)],
};

function blankCell(index) {
  return {
    id: `cell_${index + 1}`,
    model: models[0] ? models[0].id : "custom-hf-repo",
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
        el("td", {}, model.hf_repo ? `${model.hf_repo}${model.subfolder ? `/${model.subfolder}` : ""}` : model.id),
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
      select(models.map((model) => [model.id, modelLabel(model)]), cell.model, (value) => {
        cell.model = value;
        renderCells();
        renderValidation();
      })
    ),
    licenseLine(entry),
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

function renderValidation() {
  const problems = validateState(state);
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
  window.__decisionSwarm = { state, toManifest, validateState };
}

async function loadModels() {
  try {
    models = await listModels();
    renderModelsTable();
    for (const cell of state.cells) {
      // the sentinel is a user choice, not a default
      if (cell.model === "custom-hf-repo" || !models.some((model) => model.id === cell.model)) {
        cell.model = models[0].id;
      }
    }
    renderAll();
  } catch (error) {
    $("#models-status").textContent = `Failed to load models: ${error}`;
  }
}

boot();
loadModels();
