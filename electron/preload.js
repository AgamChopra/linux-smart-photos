const { contextBridge, ipcRenderer } = require("electron");
const { pathToFileURL } = require("url");

function readAdditionalArg(prefix) {
  const match = process.argv.find((entry) => entry.startsWith(prefix));
  if (!match) {
    return "";
  }
  return match.slice(prefix.length);
}

const apiBaseUrl = readAdditionalArg("--smart-photos-api-base-url=");

contextBridge.exposeInMainWorld("smartPhotos", {
  apiBaseUrl,
  toFileUrl(filePath) {
    return filePath ? pathToFileURL(filePath).href : "";
  },
  showItemInFolder(filePath) {
    return ipcRenderer.invoke("smart-photos:show-item-in-folder", filePath);
  },
  openPath(filePath) {
    return ipcRenderer.invoke("smart-photos:open-path", filePath);
  },
  resolveApiBaseUrl() {
    return ipcRenderer.invoke("smart-photos:api-base-url");
  },
});
