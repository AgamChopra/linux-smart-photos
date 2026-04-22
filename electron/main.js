const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const path = require("path");
const { spawn } = require("child_process");

let mainWindow = null;
let apiProcess = null;
let apiBaseUrl = "";
let apiReady = null;

function projectRoot() {
  return path.resolve(__dirname, "..");
}

function pythonExecutable() {
  return (
    process.env.SMART_PHOTOS_VENV_PYTHON ||
    path.join(projectRoot(), ".venv", "bin", "python")
  );
}

function extractConfigArg() {
  const args = process.argv.slice(1);
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--config" && args[index + 1]) {
      return args[index + 1];
    }
  }
  return "";
}

function startApiServer() {
  if (apiReady) {
    return apiReady;
  }

  apiReady = new Promise((resolve, reject) => {
    const args = ["-m", "linux_smart_photos.web_api", "--port", "0", "--startup-sync"];
    const configPath = extractConfigArg();
    if (configPath) {
      args.push("--config", configPath);
    }

    apiProcess = spawn(pythonExecutable(), args, {
      cwd: projectRoot(),
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdoutBuffer = "";
    let resolved = false;

    const tryResolve = (chunk) => {
      stdoutBuffer += chunk.toString();
      const lines = stdoutBuffer.split(/\r?\n/);
      stdoutBuffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.startsWith("SMART_PHOTOS_API_PORT=")) {
          continue;
        }
        const port = line.split("=", 2)[1];
        apiBaseUrl = `http://127.0.0.1:${port}`;
        resolved = true;
        resolve(apiBaseUrl);
      }
    };

    apiProcess.stdout.on("data", tryResolve);
    apiProcess.stderr.on("data", (chunk) => {
      process.stderr.write(chunk);
    });

    apiProcess.on("exit", (code, signal) => {
      if (resolved) {
        return;
      }
      reject(new Error(`Smart Photos API exited before startup (code=${code}, signal=${signal})`));
    });
  });

  return apiReady;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1020,
    minWidth: 1200,
    minHeight: 760,
    backgroundColor: "#f4ede5",
    title: "Smart Photos",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      additionalArguments: [`--smart-photos-api-base-url=${apiBaseUrl}`],
    },
  });

  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function stopApiServer() {
  if (!apiProcess) {
    return;
  }
  const child = apiProcess;
  apiProcess = null;
  try {
    child.kill();
  } catch (_error) {
    // Ignore process shutdown errors during app exit.
  }
}

ipcMain.handle("smart-photos:api-base-url", () => apiBaseUrl);
ipcMain.handle("smart-photos:show-item-in-folder", (_event, filePath) => {
  if (!filePath) {
    return false;
  }
  shell.showItemInFolder(filePath);
  return true;
});
ipcMain.handle("smart-photos:open-path", async (_event, filePath) => {
  if (!filePath) {
    return false;
  }
  await shell.openPath(filePath);
  return true;
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  stopApiServer();
});

app.whenReady().then(async () => {
  try {
    await startApiServer();
    createWindow();
  } catch (error) {
    dialog.showErrorBox("Smart Photos", `${error}`);
    app.quit();
    return;
  }

  app.on("activate", () => {
    if (!BrowserWindow.getAllWindows().length) {
      createWindow();
    }
  });
});
