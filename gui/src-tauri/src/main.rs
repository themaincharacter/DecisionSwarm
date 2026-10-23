// Decision Swarm GUI — thin Tauri shell.
//
// All pack logic lives in the decision-swarm Python package; the shell drives
// it as a CLI with JSON output. Set DECISION_SWARM_CLI to point at the
// PyInstaller launcher (or any decision-swarm entrypoint); the bundled app
// ships that launcher as a sidecar.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::process::Command;

fn cli_program() -> String {
    std::env::var("DECISION_SWARM_CLI").unwrap_or_else(|_| "decision-swarm".to_string())
}

fn run_cli(args: &[String]) -> Result<String, String> {
    let output = Command::new(cli_program())
        .args(args)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", cli_program()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(stderr.trim().to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// The model dropdown source: the registry as JSON.
#[tauri::command]
fn list_models() -> Result<String, String> {
    run_cli(&["models".to_string(), "--json".to_string()])
}

/// The webview cannot write files directly; the shell does it.
#[tauri::command]
fn write_manifest(json: String, path: String) -> Result<String, String> {
    fs::write(&path, &json).map_err(|error| format!("cannot write {path}: {error}"))?;
    Ok(path)
}

/// Download button: fetch the selected models into the shared cache.
#[tauri::command]
fn fetch_models(manifest_path: String, models_dir: Option<String>) -> Result<String, String> {
    let mut args = vec!["fetch".to_string(), "--manifest".to_string(), manifest_path];
    if let Some(dir) = models_dir {
        args.push("--models-dir".to_string());
        args.push(dir);
    }
    run_cli(&args)
}

/// Submit: validate the config and write the swarm pack.
#[tauri::command]
fn build_pack(manifest_path: String, dest: String, models_dir: Option<String>) -> Result<String, String> {
    let mut args = vec![
        "build".to_string(),
        "--manifest".to_string(),
        manifest_path,
        "--dest".to_string(),
        dest,
        "--json".to_string(),
    ];
    if let Some(dir) = models_dir {
        args.push("--models-dir".to_string());
        args.push(dir);
    }
    run_cli(&args)
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            list_models,
            write_manifest,
            fetch_models,
            build_pack
        ])
        .run(tauri::generate_context!())
        .expect("error while running the Decision Swarm GUI");
}
