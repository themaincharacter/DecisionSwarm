#!/usr/bin/env node
// Copy the decision-swarm engine launcher into `src-tauri/binaries/` so the
// release build can embed it as a sidecar (`bundle > externalBin`, added by
// bundle.release.json). Tauri looks for
// `binaries/decision-swarm-<target-triple>[.exe]`.
//
// Launcher search order: $DECISION_SWARM_SIDECAR, then the PyInstaller build
// produced at the repository root by `decision-swarm.spec`. The shell finds the
// copied launcher next to the app executable at runtime, so installed apps need
// no Python and no git checkout.

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const guiDir = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const repoRoot = path.dirname(guiDir);
const exe = process.platform === "win32" ? ".exe" : "";

function targetTriple() {
  try {
    return execFileSync("rustc", ["--print", "host-tuple"], { encoding: "utf8" }).trim();
  } catch {
    try {
      const verbose = execFileSync("rustc", ["-Vv"], { encoding: "utf8" });
      const host = verbose.split("\n").find((line) => line.trim().startsWith("host:"));
      if (host) return host.trim().slice("host:".length).trim();
    } catch {
      // rustc missing entirely: fall back to a static platform table.
    }
    const arch = { x64: "x86_64", arm64: "aarch64" }[process.arch] ?? process.arch;
    const platform = {
      linux: "unknown-linux-gnu",
      darwin: "apple-darwin",
      win32: "pc-windows-msvc",
    }[process.platform];
    if (!platform) {
      console.error(`sidecar: unsupported platform ${process.platform}/${process.arch}`);
      process.exit(1);
    }
    return `${arch}-${platform}`;
  }
}

const candidates = [
  process.env.DECISION_SWARM_SIDECAR,
  path.join(repoRoot, "dist", `decision-swarm${exe}`),
].filter(Boolean);

const source = candidates.find((candidate) => fs.existsSync(candidate));

if (!source) {
  console.error(
    [
      "sidecar: decision-swarm engine launcher not found.",
      "Build it first, from the repository root:",
      "  python -m pip install pyinstaller -e .",
      "  python -m pyinstaller --clean --noconfirm decision-swarm.spec",
      "or point DECISION_SWARM_SIDECAR at an existing launcher.",
    ].join("\n")
  );
  process.exit(1);
}

const destDir = path.join(guiDir, "src-tauri", "binaries");
fs.mkdirSync(destDir, { recursive: true });
const dest = path.join(destDir, `decision-swarm-${targetTriple()}${exe}`);
fs.copyFileSync(source, dest);
if (process.platform !== "win32") {
  fs.chmodSync(dest, 0o755);
}
console.log(`sidecar: ${source} -> ${dest}`);
