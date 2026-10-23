// Decision Swarm GUI — thin Tauri shell.
//
// All pack logic lives in the decision-swarm Python package; the shell drives
// it as a CLI with JSON output. Set DECISION_SWARM_CLI to point at the
// PyInstaller launcher (or any decision-swarm entrypoint); the bundled app
// ships that launcher as a sidecar.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use tauri::Emitter;

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

fn write_temp_manifest(json: &str) -> Result<String, String> {
    let path = std::env::temp_dir().join(format!("decision-swarm-manifest-{}.json", std::process::id()));
    fs::write(&path, json).map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(path.display().to_string())
}

/// The model dropdown source: the registry as JSON.
#[tauri::command]
fn list_models() -> Result<String, String> {
    run_cli(&["models".to_string(), "--json".to_string()])
}

/// Download button: fetch the selected models into the shared cache,
/// streaming each engine line to the UI as an `engine-line` event.
#[tauri::command]
fn download_models(
    app: tauri::AppHandle,
    manifest_json: String,
    models_dir: Option<String>,
) -> Result<String, String> {
    let manifest_path = write_temp_manifest(&manifest_json)?;
    let mut args = vec!["fetch".to_string(), "--manifest".to_string(), manifest_path];
    if let Some(dir) = models_dir {
        args.push("--models-dir".to_string());
        args.push(dir);
    }
    let mut child = Command::new(cli_program())
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to run {}: {error}", cli_program()))?;
    let stdout = child.stdout.take().ok_or_else(|| "no stdout".to_string())?;
    let mut collected = String::new();
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        let _ = app.emit("engine-line", line.clone());
        collected.push_str(&line);
        collected.push('\n');
    }
    let output = child.wait_with_output().map_err(|error| error.to_string())?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(stderr.trim().to_string());
    }
    Ok(collected)
}

/// Submit: validate the config and write the swarm pack.
#[tauri::command]
fn submit_pack(
    manifest_json: String,
    dest: String,
    models_dir: Option<String>,
) -> Result<String, String> {
    let manifest_path = write_temp_manifest(&manifest_json)?;
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
            download_models,
            submit_pack
        ])
        .run(tauri::generate_context!())
        .expect("error while running the Decision Swarm GUI");
}
