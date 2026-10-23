// Decision Swarm GUI — thin Tauri shell.
//
// All pack logic lives in the decision-swarm Python package; the shell drives
// it as a CLI with JSON output. Set DECISION_SWARM_CLI to point at the
// PyInstaller launcher (or any decision-swarm entrypoint); release bundles
// ship that launcher as a sidecar (bundle > externalBin, added on top of
// tauri.conf.json by bundle.release.json at build time).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tauri::Emitter;

fn cli_names_in(dir: &Path) -> Vec<PathBuf> {
    vec![dir.join("decision-swarm"), dir.join("decision-swarm.exe")]
}

/// Directories a bundled sidecar can live in: beside the app executable
/// (Windows installers, Linux packages, macOS ``Contents/MacOS``) plus the
/// macOS ``Contents/Resources`` fallback.
fn bundle_dirs(exe_dir: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![exe_dir.to_path_buf()];
    if let Some(parent) = exe_dir.parent() {
        dirs.push(parent.join("Resources"));
    }
    dirs.push(exe_dir.join("resources"));
    dirs
}

fn dev_cli_candidates(root: &Path) -> Vec<PathBuf> {
    vec![
        root.join(".venv/bin/decision-swarm"),
        root.join(".venv/Scripts/decision-swarm.exe"),
        root.join("dist/decision-swarm"),
        root.join("dist/decision-swarm.exe"),
    ]
}

/// Walk up from `start` looking for a repo checkout with a built engine.
fn find_dev_cli(start: &Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        for candidate in dev_cli_candidates(ancestor) {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn find_on_path() -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&paths) {
        for candidate in cli_names_in(&dir) {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Engine lookup: explicit override > launcher bundled beside the app >
/// repo dev checkout (`npm run dev` / `cargo run`) > PATH.
fn cli_program() -> Result<String, String> {
    if let Ok(program) = std::env::var("DECISION_SWARM_CLI") {
        return Ok(program);
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(dir) = executable.parent() {
            for search in bundle_dirs(dir) {
                for candidate in cli_names_in(&search) {
                    if candidate.is_file() {
                        return Ok(candidate.display().to_string());
                    }
                }
            }
        }
        if let Some(found) = find_dev_cli(&executable) {
            return Ok(found.display().to_string());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        if let Some(found) = find_dev_cli(&cwd) {
            return Ok(found.display().to_string());
        }
    }
    if let Some(found) = find_on_path() {
        return Ok(found.display().to_string());
    }
    Err(
        "decision-swarm engine not found: install it on PATH, set DECISION_SWARM_CLI to the \
         launcher, or run from a repo checkout with a .venv"
            .to_string(),
    )
}

fn run_cli(args: &[String]) -> Result<String, String> {
    let program = cli_program()?;
    let output = Command::new(&program)
        .args(args)
        .output()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
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
    let program = cli_program()?;
    let mut child = Command::new(&program)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_repo_venv_cli_from_a_nested_debug_build() {
        let root = std::env::temp_dir().join("decision-swarm-cli-test");
        let _ = fs::remove_dir_all(&root);
        let venv_bin = root.join(".venv").join("bin");
        fs::create_dir_all(&venv_bin).unwrap();
        let cli = venv_bin.join("decision-swarm");
        fs::write(&cli, b"#!/bin/sh\n").unwrap();

        let nested = root.join("gui/src-tauri/target/debug");
        fs::create_dir_all(&nested).unwrap();
        assert_eq!(find_dev_cli(&nested.join("decision-swarm-gui")), Some(cli));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn finds_nothing_without_an_engine() {
        let root = std::env::temp_dir().join("decision-swarm-cli-test-empty");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        assert_eq!(find_dev_cli(&root), None);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn bundle_dirs_cover_the_app_layouts() {
        let macos = Path::new("/Applications/Decision Swarm.app/Contents/MacOS");
        assert_eq!(
            bundle_dirs(macos),
            vec![
                macos.to_path_buf(),
                Path::new("/Applications/Decision Swarm.app/Contents/Resources").to_path_buf(),
                macos.join("resources"),
            ]
        );
    }
}
