import { listModels, runningInTauri } from "./bridge.js";

const modelsStatus = document.getElementById("models-status");
const modelsBody = document.querySelector("#models-table tbody");

function checkpointLabel(model) {
  if (model.hf_repo && model.subfolder) {
    return `${model.hf_repo}/${model.subfolder}`;
  }
  return model.hf_repo || model.id;
}

function renderModels(models) {
  const rows = models.map((model) => {
    const row = document.createElement("tr");
    row.dataset.modelId = model.id;
    const values = [
      model.id,
      checkpointLabel(model),
      model.modality || "custom",
      model.params || "—",
      model.license || "unknown (user-supplied)",
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    }
    return row;
  });
  modelsBody.replaceChildren(...rows);
  modelsStatus.textContent = `${models.length} models · engine: ${runningInTauri() ? "CLI sidecar" : "browser mock"}`;
}

async function boot() {
  try {
    renderModels(await listModels());
  } catch (error) {
    modelsStatus.textContent = `Failed to load models: ${error}`;
  }
}

boot();
