use std::{
    env,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
};
use tauri::Manager;

#[derive(Default)]
struct ApiState {
    inner: Mutex<ApiProcess>,
}

#[derive(Default)]
struct ApiProcess {
    base_url: Option<String>,
    child: Option<Child>,
}

fn project_root() -> PathBuf {
    if let Ok(root) = env::var("SMART_PHOTOS_PROJECT_ROOT") {
        return PathBuf::from(root);
    }
    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn python_executable(root: &Path) -> PathBuf {
    if let Ok(python) = env::var("SMART_PHOTOS_VENV_PYTHON") {
        return PathBuf::from(python);
    }
    root.join(".venv").join("bin").join("python")
}

fn start_api_locked(process: &mut ApiProcess) -> Result<String, String> {
    if let Some(base_url) = &process.base_url {
        return Ok(base_url.clone());
    }

    let root = project_root();
    let python = python_executable(&root);
    let mut args = vec![
        "-m".to_string(),
        "linux_smart_photos.web_api".to_string(),
        "--port".to_string(),
        "0".to_string(),
        "--startup-sync".to_string(),
    ];
    let cli_args: Vec<String> = env::args().collect();
    for index in 0..cli_args.len() {
        if cli_args[index] == "--config" {
            if let Some(config_path) = cli_args.get(index + 1) {
                args.push("--config".to_string());
                args.push(config_path.clone());
            }
        }
    }
    let mut child = Command::new(python)
        .args(args)
        .current_dir(&root)
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("Unable to start LSP API: {error}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "LSP API stdout was unavailable.".to_string())?;
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let port = loop {
        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|error| format!("Unable to read LSP API startup output: {error}"))?;
        if bytes == 0 {
            return Err("LSP API exited before reporting its port.".to_string());
        }
        if let Some(value) = line.strip_prefix("SMART_PHOTOS_API_PORT=") {
            break value.trim().to_string();
        }
    };

    let base_url = format!("http://127.0.0.1:{port}");
    process.base_url = Some(base_url.clone());
    process.child = Some(child);
    Ok(base_url)
}

#[tauri::command]
fn api_base_url(state: tauri::State<ApiState>) -> Result<String, String> {
    let mut process = state
        .inner
        .lock()
        .map_err(|_| "LSP API state lock failed.".to_string())?;
    start_api_locked(&mut process)
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    if path.trim().is_empty() {
        return Ok(());
    }
    Command::new("xdg-open")
        .arg(path)
        .spawn()
        .map_err(|error| format!("Unable to open path: {error}"))?;
    Ok(())
}

#[tauri::command]
fn show_item_in_folder(path: String) -> Result<(), String> {
    if path.trim().is_empty() {
        return Ok(());
    }
    let target = PathBuf::from(path);
    let folder = target.parent().unwrap_or_else(|| Path::new("."));
    Command::new("xdg-open")
        .arg(folder)
        .spawn()
        .map_err(|error| format!("Unable to open folder: {error}"))?;
    Ok(())
}

fn stop_api(state: &ApiState) {
    if let Ok(mut process) = state.inner.lock() {
        if let Some(mut child) = process.child.take() {
            let _ = child.kill();
        }
        process.base_url = None;
    }
}

fn main() {
    tauri::Builder::default()
        .manage(ApiState::default())
        .invoke_handler(tauri::generate_handler![
            api_base_url,
            open_path,
            show_item_in_folder
        ])
        .build(tauri::generate_context!())
        .expect("failed to build Linux Smart Photos Tauri app")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(state) = app_handle.try_state::<ApiState>() {
                    stop_api(&state);
                }
            }
        });
}
