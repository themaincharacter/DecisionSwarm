// Bridge between the web UI and the Decision Swarm engine.
//
// Inside Tauri the UI invokes Rust commands, which drive the decision-swarm
// CLI with JSON output. In a plain browser a deterministic mock keeps every
// screen exercisable — the UI never assumes Tauri.

const isTauri = typeof window !== "undefined" && Boolean(window.__TAURI_INTERNALS__);

const MOCK_MODELS = [
  {
    id: "laya-english",
    hf_repo: "convaiinnovations/laya",
    subfolder: null,
    revision: "main",
    modality: "text",
    package_source: "laya>=0.3.20",
    loader: "load",
    params: "421M",
    download_bytes: 842609210,
    license: "Apache-2.0",
    custom: false,
  },
  {
    id: "laya-multilingual",
    hf_repo: "convaiinnovations/laya",
    subfolder: "multilingual",
    revision: "main",
    modality: "text",
    package_source: "laya>=0.3.20",
    loader: "load",
    params: "322M",
    download_bytes: 643835514,
    license: "Apache-2.0",
    custom: false,
  },
  {
    id: "laya-typed-decisions",
    hf_repo: "convaiinnovations/laya",
    subfolder: "typed-decisions",
    revision: "main",
    modality: "text",
    package_source: "laya>=0.3.20",
    loader: "load",
    params: "421M",
    download_bytes: 842609220,
    license: "Apache-2.0",
    custom: false,
  },
  {
    id: "laya-vision",
    hf_repo: "thaitea/laya-vision",
    subfolder: null,
    revision: "main",
    modality: "image+text",
    package_source: "git+https://github.com/r33drichards/laya-vision",
    loader: "load_vlm",
    params: "201M",
    download_bytes: 804000000,
    license: "weights: CC BY-NC-SA 4.0 / code: Apache-2.0",
    custom: false,
  },
  {
    id: "laya-vision-237m",
    hf_repo: "thaitea/laya-vision-smolvlm-256m-score",
    subfolder: null,
    revision: "main",
    modality: "image+text",
    package_source: "git+https://github.com/r33drichards/laya-vision",
    loader: "load_vlm",
    params: "237M",
    download_bytes: 948000000,
    license: "weights: CC BY-NC-SA 4.0 / code: Apache-2.0",
    custom: false,
  },
  {
    id: "custom-hf-repo",
    hf_repo: null,
    subfolder: null,
    revision: null,
    modality: null,
    package_source: null,
    loader: null,
    params: null,
    download_bytes: null,
    license: null,
    custom: true,
  },
];

async function invoke(command, args = {}) {
  return window.__TAURI__.core.invoke(command, args);
}

export async function listModels() {
  if (isTauri) {
    const raw = await invoke("list_models");
    return JSON.parse(raw).models;
  }
  return MOCK_MODELS;
}

export async function downloadModels(manifestJson, modelsDir = null, onLine = () => {}) {
  if (isTauri) {
    const unlisten = await window.__TAURI__.event.listen("engine-line", (event) => onLine(event.payload));
    try {
      return await invoke("download_models", { manifestJson, modelsDir });
    } finally {
      unlisten();
    }
  }
  const manifest = JSON.parse(manifestJson);
  const ids = [...new Set(manifest.cells.map((cell) => cell.model))];
  for (const id of ids) {
    await new Promise((resolve) => setTimeout(resolve, 250));
    onLine(`${id}  downloaded  (into the shared cache)`);
  }
  return "ok";
}

export async function submitPack(manifestJson, dest, modelsDir = null) {
  if (isTauri) {
    const raw = await invoke("submit_pack", { manifestJson, dest, modelsDir });
    return JSON.parse(raw);
  }
  await new Promise((resolve) => setTimeout(resolve, 250));
  const manifest = JSON.parse(manifestJson);
  return {
    pack: dest,
    name: manifest.name,
    cells: manifest.cells.map((cell) => cell.id),
    models: [],
  };
}

export function runningInTauri() {
  return isTauri;
}
