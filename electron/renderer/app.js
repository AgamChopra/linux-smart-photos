const apiBaseUrl =
  window.smartPhotos.apiBaseUrl || (await window.smartPhotos.resolveApiBaseUrl());

const state = {
  activeView: "library",
  status: null,
  items: [],
  selectedItem: null,
  selectedDetectionId: "",
  personas: [],
  selectedPersonaId: "",
  unknownClusters: [],
  selectedUnknownClusterIds: new Set(),
  albums: [],
  selectedAlbumId: "",
  memories: [],
  selectedMemoryId: "",
  libraryRefreshTimer: null,
  lastUpdatedAt: "",
};

const elements = {
  navButtons: [...document.querySelectorAll(".nav-button")],
  viewSections: [...document.querySelectorAll(".view-section")],
  viewTitle: document.querySelector("#view-title"),
  viewSubtitle: document.querySelector("#view-subtitle"),
  countItems: document.querySelector("#count-items"),
  countPersonas: document.querySelector("#count-personas"),
  countAlbums: document.querySelector("#count-albums"),
  countMemories: document.querySelector("#count-memories"),
  statusPill: document.querySelector("#status-pill"),
  statusText: document.querySelector("#status-text"),
  progressBar: document.querySelector("#progress-bar"),
  syncButton: document.querySelector("#sync-button"),
  openFolderButton: document.querySelector("#open-folder-button"),
  librarySearch: document.querySelector("#library-search"),
  libraryType: document.querySelector("#library-type"),
  libraryPersonaKind: document.querySelector("#library-persona-kind"),
  libraryPersona: document.querySelector("#library-persona"),
  libraryFavorites: document.querySelector("#library-favorites"),
  libraryGrid: document.querySelector("#library-grid"),
  libraryResultsCount: document.querySelector("#library-results-count"),
  itemPreview: document.querySelector("#item-preview"),
  itemDetails: document.querySelector("#item-details"),
  itemFavoriteButton: document.querySelector("#item-favorite-button"),
  itemOpenFileButton: document.querySelector("#item-open-file-button"),
  itemOpenFolderDetailButton: document.querySelector("#item-open-folder-detail-button"),
  itemDetections: document.querySelector("#item-detections"),
  assignKind: document.querySelector("#assign-kind"),
  assignPersona: document.querySelector("#assign-persona"),
  assignNewName: document.querySelector("#assign-new-name"),
  assignRegionButton: document.querySelector("#assign-region-button"),
  clearRegionButton: document.querySelector("#clear-region-button"),
  assignItemButton: document.querySelector("#assign-item-button"),
  clearItemButton: document.querySelector("#clear-item-button"),
  albumSelect: document.querySelector("#album-select"),
  albumNewName: document.querySelector("#album-new-name"),
  albumAddButton: document.querySelector("#album-add-button"),
  personaKindFilter: document.querySelector("#persona-kind-filter"),
  newPersonaKind: document.querySelector("#new-persona-kind"),
  newPersonaName: document.querySelector("#new-persona-name"),
  newPersonaButton: document.querySelector("#new-persona-button"),
  personaList: document.querySelector("#persona-list"),
  personaDetailTitle: document.querySelector("#persona-detail-title"),
  personaReferenceStrip: document.querySelector("#persona-reference-strip"),
  personaGrid: document.querySelector("#persona-grid"),
  unknownKind: document.querySelector("#unknown-kind"),
  unknownAssignKind: document.querySelector("#unknown-assign-kind"),
  unknownAssignPersona: document.querySelector("#unknown-assign-persona"),
  unknownNewName: document.querySelector("#unknown-new-name"),
  unknownAssignButton: document.querySelector("#unknown-assign-button"),
  unknownList: document.querySelector("#unknown-list"),
  unknownSummary: document.querySelector("#unknown-summary"),
  unknownGrid: document.querySelector("#unknown-grid"),
  newAlbumName: document.querySelector("#new-album-name"),
  newAlbumButton: document.querySelector("#new-album-button"),
  deleteAlbumButton: document.querySelector("#delete-album-button"),
  albumList: document.querySelector("#album-list"),
  albumDetailTitle: document.querySelector("#album-detail-title"),
  albumGrid: document.querySelector("#album-grid"),
  memoryList: document.querySelector("#memory-list"),
  memoryDetailTitle: document.querySelector("#memory-detail-title"),
  memorySummary: document.querySelector("#memory-summary"),
  memoryGrid: document.querySelector("#memory-grid"),
  modelList: document.querySelector("#model-list"),
  downloadModelsButton: document.querySelector("#download-models-button"),
};

const viewCopy = {
  library: {
    title: "Library",
    subtitle: "Browse the indexed library and correct people, pets, and albums.",
  },
  personas: {
    title: "People & Pets",
    subtitle: "Manage named personas, references, and grouped items.",
  },
  unknown: {
    title: "Unknown Clusters",
    subtitle: "Review unassigned face and pet clusters, then merge them into personas.",
  },
  albums: {
    title: "Albums",
    subtitle: "Build and curate custom collections.",
  },
  memories: {
    title: "Memories",
    subtitle: "Browse automatically generated memory reels.",
  },
  models: {
    title: "AI Models",
    subtitle: "Inspect model availability and download the recommended set.",
  },
};

async function fetchJson(path, options = {}) {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    const message = payload?.error || `Request failed: ${response.status}`;
    throw new Error(message);
  }
  return payload;
}

function debounce(callback, delay = 280) {
  return (...args) => {
    if (state.libraryRefreshTimer) {
      clearTimeout(state.libraryRefreshTimer);
    }
    state.libraryRefreshTimer = setTimeout(() => callback(...args), delay);
  };
}

function itemPreviewPath(item) {
  if (!item) {
    return "";
  }
  if (item.mediaKind === "video") {
    return item.componentPaths?.[0] || item.path;
  }
  if (item.mediaKind === "live_photo") {
    const videoPath = (item.componentPaths || []).find((entry) =>
      entry.match(/\.(mov|mp4|m4v)$/i),
    );
    return videoPath || item.path;
  }
  return item.path;
}

function itemPreviewKind(item) {
  if (!item) {
    return "";
  }
  if (item.mediaKind === "video" || item.mediaKind === "live_photo") {
    return "video";
  }
  return "image";
}

function setEmpty(container, message) {
  container.innerHTML = `<div class="muted-text">${message}</div>`;
}

function renderMediaGrid(container, items, selectedId, onSelect) {
  if (!items.length) {
    setEmpty(container, "No items to show.");
    return;
  }
  container.innerHTML = "";
  for (const item of items) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `media-card${item.id === selectedId ? " selected" : ""}`;
    const thumbSrc = item.thumbnailPath ? window.smartPhotos.toFileUrl(item.thumbnailPath) : "";
    card.innerHTML = `
      ${
        thumbSrc
          ? `<img class="media-thumb" src="${thumbSrc}" alt="" />`
          : `<div class="media-thumb"></div>`
      }
      <div class="media-meta">
        <div class="media-title">${escapeHtml(item.title)}</div>
        <div class="media-caption">${escapeHtml(item.mediaKind)} • ${escapeHtml((item.capturedAt || "").slice(0, 10))}</div>
        <div class="tag-list">${(item.personaNames || []).slice(0, 3).map((name) => `<span class="tag-chip">${escapeHtml(name)}</span>`).join("")}</div>
      </div>
    `;
    card.addEventListener("click", () => onSelect(item.id));
    container.appendChild(card);
  }
}

function renderSimpleList(container, entries, selectedId, onSelect, options = {}) {
  if (!entries.length) {
    setEmpty(container, options.emptyMessage || "Nothing here yet.");
    return;
  }
  container.innerHTML = "";
  for (const entry of entries) {
    const element = document.createElement("button");
    element.type = "button";
    element.className = `list-item${selectedId === entry.id ? " selected" : ""}`;
    element.innerHTML = `
      <div class="list-item-title">${escapeHtml(entry.title)}</div>
      <div class="list-item-subtitle">${escapeHtml(entry.subtitle || "")}</div>
    `;
    element.addEventListener("click", () => onSelect(entry.id));
    container.appendChild(element);
  }
}

function renderSelectableList(container, entries, selectedIds, onToggle, options = {}) {
  if (!entries.length) {
    setEmpty(container, options.emptyMessage || "Nothing here yet.");
    return;
  }
  container.innerHTML = "";
  for (const entry of entries) {
    const element = document.createElement("button");
    element.type = "button";
    element.className = `list-item${selectedIds.has(entry.id) ? " selected" : ""}`;
    element.innerHTML = `
      <div class="list-item-title">${escapeHtml(entry.title)}</div>
      <div class="list-item-subtitle">${escapeHtml(entry.subtitle || "")}</div>
    `;
    element.addEventListener("click", () => onToggle(entry.id));
    container.appendChild(element);
  }
}

function renderDetections(item) {
  if (!item || !item.detections?.length) {
    setEmpty(elements.itemDetections, "No detected faces or pets on this item.");
    return;
  }
  elements.itemDetections.innerHTML = "";
  for (const detection of item.detections) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `detection-row${state.selectedDetectionId === detection.id ? " selected" : ""}`;
    row.innerHTML = `
      <div class="detection-title">${escapeHtml(detection.label)} • ${escapeHtml(detection.kind)}</div>
      <div class="detection-subtitle">Confidence ${Number(detection.confidence).toFixed(2)} • ${
        detection.personaId ? `Assigned to ${escapeHtml(personaNameById(detection.personaId))}` : "Unassigned"
      }</div>
    `;
    row.addEventListener("click", () => {
      state.selectedDetectionId = detection.id;
      if (detection.kind === "face") {
        elements.assignKind.value = "person";
      } else if (detection.kind.startsWith("pet") || ["cat", "dog", "pet"].includes(String(detection.label).toLowerCase())) {
        elements.assignKind.value = "pet";
      }
      syncPersonaSelects();
      renderDetections(state.selectedItem);
    });
    elements.itemDetections.appendChild(row);
  }
}

function renderSelectedItem(item) {
  state.selectedItem = item;
  state.selectedDetectionId = state.selectedDetectionId && item?.detections?.some((entry) => entry.id === state.selectedDetectionId)
    ? state.selectedDetectionId
    : item?.detections?.[0]?.id || "";

  if (!item) {
    elements.itemPreview.classList.add("empty");
    elements.itemPreview.innerHTML = "Select an item to preview it.";
    elements.itemDetails.textContent = "No item selected.";
    renderDetections(null);
    return;
  }

  const previewPath = itemPreviewPath(item);
  const previewKind = itemPreviewKind(item);
  elements.itemPreview.classList.remove("empty");
  elements.itemPreview.innerHTML =
    previewKind === "video"
      ? `<video controls src="${window.smartPhotos.toFileUrl(previewPath)}"></video>`
      : `<img src="${window.smartPhotos.toFileUrl(previewPath)}" alt="" />`;
  elements.itemDetails.textContent = item.details || "No details available.";
  renderDetections(item);
}

function renderPersonasList(personas) {
  renderSimpleList(
    elements.personaList,
    personas.map((persona) => ({
      id: persona.id,
      title: persona.name,
      subtitle: `${persona.kind} • ${persona.referenceImageCount} references`,
    })),
    state.selectedPersonaId,
    async (personaId) => {
      state.selectedPersonaId = personaId;
      await loadSelectedPersona();
    },
    { emptyMessage: "No personas yet." },
  );
}

function renderReferenceStrip(references) {
  if (!references.length) {
    setEmpty(elements.personaReferenceStrip, "No reference crops yet.");
    return;
  }
  elements.personaReferenceStrip.innerHTML = "";
  for (const reference of references) {
    const card = document.createElement("div");
    card.className = "reference-card";
    card.innerHTML = `
      <img src="${window.smartPhotos.toFileUrl(reference.path)}" alt="" />
      <div class="reference-label">${escapeHtml(reference.label || reference.kind || "reference")}</div>
    `;
    elements.personaReferenceStrip.appendChild(card);
  }
}

function renderModels(models) {
  if (!models.length) {
    setEmpty(elements.modelList, "No model metadata found.");
    return;
  }
  elements.modelList.innerHTML = "";
  for (const model of models) {
    const row = document.createElement("div");
    row.className = "model-row";
    row.innerHTML = `
      <div class="model-title-row">
        <strong>${escapeHtml(model.title)}</strong>
        <span class="${model.installed ? "installed-pill" : "missing-pill"}">
          ${model.installed ? "Installed" : "Missing"}
        </span>
      </div>
      <div class="model-subtitle">${escapeHtml(model.description || "")}</div>
      <div class="model-subtitle">${escapeHtml(model.localPath || "")}</div>
    `;
    elements.modelList.appendChild(row);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function personaNameById(personaId) {
  const persona = state.personas.find((entry) => entry.id === personaId);
  return persona ? persona.name : "persona";
}

function syncPersonaSelects() {
  const kind = elements.assignKind.value;
  populatePersonaSelect(elements.assignPersona, kind, true);
  populatePersonaSelect(elements.unknownAssignPersona, elements.unknownAssignKind.value, true);
  populatePersonaSelect(elements.libraryPersona, elements.libraryPersonaKind.value, true, "Any persona");
}

function populatePersonaSelect(select, kind, allowEmpty, emptyLabel = "Existing persona") {
  const selectedValue = select.value;
  select.innerHTML = "";
  if (allowEmpty) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = emptyLabel;
    select.appendChild(option);
  }
  for (const persona of state.personas) {
    if (kind !== "all" && persona.kind !== kind) {
      continue;
    }
    const option = document.createElement("option");
    option.value = persona.id;
    option.textContent = `${persona.name} (${persona.kind})`;
    select.appendChild(option);
  }
  if ([...select.options].some((option) => option.value === selectedValue)) {
    select.value = selectedValue;
  }
}

function populateAlbumSelect() {
  const selectedValue = elements.albumSelect.value;
  elements.albumSelect.innerHTML = `<option value="">Existing album</option>`;
  for (const album of state.albums) {
    const option = document.createElement("option");
    option.value = album.id;
    option.textContent = album.name;
    elements.albumSelect.appendChild(option);
  }
  if ([...elements.albumSelect.options].some((option) => option.value === selectedValue)) {
    elements.albumSelect.value = selectedValue;
  }
}

async function loadStatus() {
  const status = await fetchJson("/api/status");
  const previousUpdatedAt = state.lastUpdatedAt;
  state.status = status;
  state.lastUpdatedAt = status.updatedAt || "";

  elements.countItems.textContent = status.counts.items;
  elements.countPersonas.textContent = status.counts.personas;
  elements.countAlbums.textContent = status.counts.albums;
  elements.countMemories.textContent = status.counts.memories;
  elements.statusPill.textContent = status.job?.status === "running" ? "Working" : "Ready";
  const detail = status.job?.detail ? ` • ${status.job.detail}` : "";
  elements.statusText.textContent = status.job
    ? `${status.job.message}${detail}`
    : "Ready.";

  if (status.job?.status === "running" && status.job.total > 0) {
    const percent = Math.max(2, Math.min(100, Math.round((status.job.current / status.job.total) * 100)));
    elements.progressBar.style.width = `${percent}%`;
  } else if (status.job?.status === "running") {
    elements.progressBar.style.width = "45%";
  } else {
    elements.progressBar.style.width = "0%";
  }

  renderModels(status.models || []);

  if (state.activeView === "library" && previousUpdatedAt && previousUpdatedAt !== state.lastUpdatedAt) {
    await loadLibrary(true);
  }
  if (state.activeView === "personas" && previousUpdatedAt && previousUpdatedAt !== state.lastUpdatedAt) {
    await loadPersonas(true);
  }
  if (state.activeView === "albums" && previousUpdatedAt && previousUpdatedAt !== state.lastUpdatedAt) {
    await loadAlbums(true);
  }
  if (state.activeView === "memories" && previousUpdatedAt && previousUpdatedAt !== state.lastUpdatedAt) {
    await loadMemories(true);
  }
  if (state.activeView === "unknown" && previousUpdatedAt && previousUpdatedAt !== state.lastUpdatedAt) {
    await loadUnknownClusters(true);
  }
}

async function loadLibrary(keepSelection = false) {
  const params = new URLSearchParams({
    query: elements.librarySearch.value.trim(),
    type: elements.libraryType.value,
    personaKind: elements.libraryPersonaKind.value,
    personaId: elements.libraryPersona.value,
    favorites: elements.libraryFavorites.checked ? "1" : "0",
    limit: "180",
  });
  const payload = await fetchJson(`/api/items?${params.toString()}`);
  state.items = payload.items || [];
  if (!keepSelection || !state.selectedItem || !state.items.some((item) => item.id === state.selectedItem.id)) {
    state.selectedItem = null;
  }
  elements.libraryResultsCount.textContent = `${state.items.length} items`;
  renderMediaGrid(elements.libraryGrid, state.items, state.selectedItem?.id || "", selectLibraryItem);
  if (state.selectedItem) {
    await selectLibraryItem(state.selectedItem.id);
  } else {
    renderSelectedItem(null);
  }
}

async function selectLibraryItem(itemId) {
  const payload = await fetchJson(`/api/items/${encodeURIComponent(itemId)}`);
  renderSelectedItem(payload);
  renderMediaGrid(elements.libraryGrid, state.items, itemId, selectLibraryItem);
}

async function openItemInLibrary(itemId) {
  setView("library");
  elements.librarySearch.value = "";
  elements.libraryType.value = "all";
  elements.libraryPersonaKind.value = "all";
  elements.libraryPersona.value = "";
  elements.libraryFavorites.checked = false;
  syncPersonaSelects();
  await loadLibrary(false);
  await selectLibraryItem(itemId);
}

async function loadPersonas(keepSelection = false) {
  const payload = await fetchJson("/api/personas?kind=all");
  state.personas = payload.personas || [];
  syncPersonaSelects();
  const filteredPersonas = state.personas.filter(
    (persona) => elements.personaKindFilter.value === "all" || persona.kind === elements.personaKindFilter.value,
  );
  if (!keepSelection || !state.selectedPersonaId || !filteredPersonas.some((persona) => persona.id === state.selectedPersonaId)) {
    state.selectedPersonaId = filteredPersonas[0]?.id || "";
  }
  renderPersonasList(filteredPersonas);
  await loadSelectedPersona();
}

async function loadSelectedPersona() {
  if (!state.selectedPersonaId) {
    elements.personaDetailTitle.textContent = "Select a persona";
    setEmpty(elements.personaReferenceStrip, "No reference crops yet.");
    setEmpty(elements.personaGrid, "No persona selected.");
    return;
  }
  const payload = await fetchJson(`/api/personas/${encodeURIComponent(state.selectedPersonaId)}`);
  elements.personaDetailTitle.textContent = `${payload.name} • ${payload.kind}`;
  renderReferenceStrip(payload.referenceImages || []);
  renderMediaGrid(elements.personaGrid, payload.items || [], "", openItemInLibrary);
}

async function loadUnknownClusters(keepSelection = false) {
  const payload = await fetchJson(`/api/unknown-clusters?kind=${encodeURIComponent(elements.unknownKind.value)}`);
  state.unknownClusters = payload.clusters || [];
  if (!keepSelection) {
    state.selectedUnknownClusterIds = new Set();
  } else {
    state.selectedUnknownClusterIds = new Set(
      [...state.selectedUnknownClusterIds].filter((clusterId) => state.unknownClusters.some((cluster) => cluster.id === clusterId)),
    );
  }
  renderUnknownList();
  await loadUnknownClusterItems();
}

function renderUnknownList() {
  renderSelectableList(
    elements.unknownList,
    state.unknownClusters.map((cluster, index) => ({
      id: cluster.id,
      title: cluster.kind === "person" ? `Unknown person ${index + 1}` : `Unknown ${cluster.label || "pet"} ${index + 1}`,
      subtitle: `${cluster.memberCount} detections • ${cluster.itemCount} items`,
    })),
    state.selectedUnknownClusterIds,
    async (clusterId) => {
      if (state.selectedUnknownClusterIds.has(clusterId)) {
        state.selectedUnknownClusterIds.delete(clusterId);
      } else {
        state.selectedUnknownClusterIds.add(clusterId);
      }
      renderUnknownList();
      await loadUnknownClusterItems();
    },
    { emptyMessage: "No unknown clusters waiting for review." },
  );
}

async function loadUnknownClusterItems() {
  const clusterIds = [...state.selectedUnknownClusterIds];
  if (!clusterIds.length) {
    elements.unknownSummary.textContent = "Select one or more clusters.";
    setEmpty(elements.unknownGrid, "No cluster items selected.");
    return;
  }
  const payload = await fetchJson("/api/unknown-clusters/items", {
    method: "POST",
    body: JSON.stringify({ clusterIds }),
  });
  const items = payload.items || [];
  const selectedClusters = state.unknownClusters.filter((cluster) => state.selectedUnknownClusterIds.has(cluster.id));
  const detectionCount = selectedClusters.reduce((sum, cluster) => sum + cluster.memberCount, 0);
  elements.unknownSummary.textContent = `${selectedClusters.length} clusters • ${detectionCount} detections • ${items.length} items`;
  renderMediaGrid(elements.unknownGrid, items, "", openItemInLibrary);
  if (selectedClusters.length === 1) {
    elements.unknownAssignKind.value = selectedClusters[0].kind;
  }
  syncPersonaSelects();
}

async function loadAlbums(keepSelection = false) {
  const payload = await fetchJson("/api/albums");
  state.albums = payload.albums || [];
  populateAlbumSelect();
  if (!keepSelection || !state.selectedAlbumId || !state.albums.some((album) => album.id === state.selectedAlbumId)) {
    state.selectedAlbumId = state.albums[0]?.id || "";
  }
  renderSimpleList(
    elements.albumList,
    state.albums.map((album) => ({
      id: album.id,
      title: album.name,
      subtitle: `${album.itemCount} items`,
    })),
    state.selectedAlbumId,
    async (albumId) => {
      state.selectedAlbumId = albumId;
      await loadSelectedAlbum();
    },
    { emptyMessage: "No albums yet." },
  );
  await loadSelectedAlbum();
}

async function loadSelectedAlbum() {
  if (!state.selectedAlbumId) {
    elements.albumDetailTitle.textContent = "Select an album";
    setEmpty(elements.albumGrid, "No album selected.");
    return;
  }
  const payload = await fetchJson(`/api/albums/${encodeURIComponent(state.selectedAlbumId)}`);
  elements.albumDetailTitle.textContent = payload.name;
  renderMediaGrid(elements.albumGrid, payload.items || [], "", openItemInLibrary);
}

async function loadMemories(keepSelection = false) {
  const payload = await fetchJson("/api/memories");
  state.memories = payload.memories || [];
  if (!keepSelection || !state.selectedMemoryId || !state.memories.some((memory) => memory.id === state.selectedMemoryId)) {
    state.selectedMemoryId = state.memories[0]?.id || "";
  }
  renderSimpleList(
    elements.memoryList,
    state.memories.map((memory) => ({
      id: memory.id,
      title: memory.title,
      subtitle: `${memory.subtitle || memory.memoryType} • ${memory.itemCount} items`,
    })),
    state.selectedMemoryId,
    async (memoryId) => {
      state.selectedMemoryId = memoryId;
      await loadSelectedMemory();
    },
    { emptyMessage: "No memories available yet." },
  );
  await loadSelectedMemory();
}

async function loadSelectedMemory() {
  if (!state.selectedMemoryId) {
    elements.memoryDetailTitle.textContent = "Select a memory";
    elements.memorySummary.textContent = "Choose a memory to view its items.";
    setEmpty(elements.memoryGrid, "No memory selected.");
    return;
  }
  const payload = await fetchJson(`/api/memories/${encodeURIComponent(state.selectedMemoryId)}`);
  elements.memoryDetailTitle.textContent = payload.title;
  elements.memorySummary.textContent = payload.summary || payload.subtitle || payload.memoryType;
  renderMediaGrid(elements.memoryGrid, payload.items || [], "", openItemInLibrary);
}

async function createPersona() {
  const name = elements.newPersonaName.value.trim();
  if (!name) {
    return;
  }
  await fetchJson("/api/personas", {
    method: "POST",
    body: JSON.stringify({
      name,
      kind: elements.newPersonaKind.value,
    }),
  });
  elements.newPersonaName.value = "";
  await loadPersonas();
  await loadStatus();
}

async function createAlbum() {
  const name = elements.newAlbumName.value.trim();
  if (!name) {
    return;
  }
  await fetchJson("/api/albums", {
    method: "POST",
    body: JSON.stringify({ name, itemIds: [] }),
  });
  elements.newAlbumName.value = "";
  await loadAlbums();
  await loadStatus();
}

async function addSelectedItemToAlbum() {
  if (!state.selectedItem) {
    return;
  }
  const albumId = elements.albumSelect.value;
  const newName = elements.albumNewName.value.trim();
  if (!albumId && !newName) {
    return;
  }

  if (newName) {
    await fetchJson("/api/albums", {
      method: "POST",
      body: JSON.stringify({ name: newName, itemIds: [state.selectedItem.id] }),
    });
    elements.albumNewName.value = "";
  } else {
    await fetchJson(`/api/albums/${encodeURIComponent(albumId)}/items`, {
      method: "POST",
      body: JSON.stringify({ itemIds: [state.selectedItem.id] }),
    });
  }
  await loadAlbums();
  await loadStatus();
}

async function deleteSelectedAlbum() {
  if (!state.selectedAlbumId) {
    return;
  }
  await fetchJson(`/api/albums/${encodeURIComponent(state.selectedAlbumId)}`, {
    method: "DELETE",
  });
  state.selectedAlbumId = "";
  await loadAlbums();
  await loadStatus();
}

async function toggleFavorite() {
  if (!state.selectedItem) {
    return;
  }
  await fetchJson(`/api/items/${encodeURIComponent(state.selectedItem.id)}/toggle-favorite`, {
    method: "POST",
    body: JSON.stringify({}),
  });
  await loadLibrary(true);
  await loadStatus();
}

async function assignSelectedRegion() {
  if (!state.selectedItem || !state.selectedDetectionId) {
    return;
  }
  await fetchJson("/api/corrections/region/assign", {
    method: "POST",
    body: JSON.stringify({
      itemId: state.selectedItem.id,
      regionId: state.selectedDetectionId,
      personaId: elements.assignPersona.value,
      newName: elements.assignNewName.value.trim(),
      kind: elements.assignKind.value,
    }),
  });
  elements.assignNewName.value = "";
  await loadPersonas(true);
  await loadLibrary(true);
  await loadStatus();
}

async function clearSelectedRegion() {
  if (!state.selectedItem || !state.selectedDetectionId) {
    return;
  }
  await fetchJson("/api/corrections/region/clear", {
    method: "POST",
    body: JSON.stringify({
      itemId: state.selectedItem.id,
      regionId: state.selectedDetectionId,
    }),
  });
  await loadLibrary(true);
  await loadStatus();
}

async function assignWholeItem() {
  if (!state.selectedItem) {
    return;
  }
  await fetchJson("/api/corrections/item/assign", {
    method: "POST",
    body: JSON.stringify({
      itemId: state.selectedItem.id,
      personaId: elements.assignPersona.value,
      newName: elements.assignNewName.value.trim(),
      kind: elements.assignKind.value,
    }),
  });
  elements.assignNewName.value = "";
  await loadPersonas(true);
  await loadLibrary(true);
  await loadStatus();
}

async function clearWholeItem() {
  if (!state.selectedItem) {
    return;
  }
  await fetchJson("/api/corrections/item/clear", {
    method: "POST",
    body: JSON.stringify({ itemId: state.selectedItem.id }),
  });
  await loadLibrary(true);
  await loadStatus();
}

async function assignUnknownClusters() {
  const clusterIds = [...state.selectedUnknownClusterIds];
  if (!clusterIds.length) {
    return;
  }
  await fetchJson("/api/unknown-clusters/assign", {
    method: "POST",
    body: JSON.stringify({
      clusterIds,
      personaId: elements.unknownAssignPersona.value,
      newName: elements.unknownNewName.value.trim(),
      kind: elements.unknownAssignKind.value,
    }),
  });
  elements.unknownNewName.value = "";
  state.selectedUnknownClusterIds = new Set();
  await loadPersonas(true);
  await loadUnknownClusters();
  await loadStatus();
}

async function triggerSync() {
  await fetchJson("/api/jobs/sync", {
    method: "POST",
    body: JSON.stringify({}),
  });
  await loadStatus();
}

async function triggerModelDownload() {
  await fetchJson("/api/jobs/models/recommended", {
    method: "POST",
    body: JSON.stringify({}),
  });
  await loadStatus();
}

function setView(viewName) {
  state.activeView = viewName;
  for (const button of elements.navButtons) {
    button.classList.toggle("active", button.dataset.view === viewName);
  }
  for (const section of elements.viewSections) {
    section.classList.toggle("active", section.dataset.view === viewName);
  }
  const copy = viewCopy[viewName];
  elements.viewTitle.textContent = copy.title;
  elements.viewSubtitle.textContent = copy.subtitle;

  if (viewName === "personas") {
    loadPersonas(true).catch(showError);
  } else if (viewName === "unknown") {
    loadUnknownClusters(true).catch(showError);
  } else if (viewName === "albums") {
    loadAlbums(true).catch(showError);
  } else if (viewName === "memories") {
    loadMemories(true).catch(showError);
  } else if (viewName === "models") {
    renderModels(state.status?.models || []);
  } else {
    loadLibrary(true).catch(showError);
  }
}

function showError(error) {
  console.error(error);
  elements.statusPill.textContent = "Error";
  elements.statusText.textContent = String(error);
}

function startStatusPolling() {
  setInterval(() => {
    loadStatus().catch(showError);
  }, 1500);
}

function bindEvents() {
  for (const button of elements.navButtons) {
    button.addEventListener("click", () => setView(button.dataset.view));
  }
  const debouncedLibraryRefresh = debounce(() => loadLibrary(false).catch(showError));
  elements.librarySearch.addEventListener("input", debouncedLibraryRefresh);
  elements.libraryType.addEventListener("change", () => loadLibrary(false).catch(showError));
  elements.libraryPersonaKind.addEventListener("change", async () => {
    syncPersonaSelects();
    await loadLibrary(false).catch(showError);
  });
  elements.libraryPersona.addEventListener("change", () => loadLibrary(false).catch(showError));
  elements.libraryFavorites.addEventListener("change", () => loadLibrary(false).catch(showError));
  elements.assignKind.addEventListener("change", syncPersonaSelects);
  elements.unknownAssignKind.addEventListener("change", syncPersonaSelects);
  elements.personaKindFilter.addEventListener("change", () => loadPersonas(false).catch(showError));
  elements.unknownKind.addEventListener("change", () => loadUnknownClusters(false).catch(showError));
  elements.syncButton.addEventListener("click", () => triggerSync().catch(showError));
  elements.openFolderButton.addEventListener("click", () => window.smartPhotos.openPath(state.status?.mediaRoot));
  elements.itemOpenFileButton.addEventListener("click", () => window.smartPhotos.openPath(state.selectedItem?.path));
  elements.itemOpenFolderDetailButton.addEventListener("click", () => window.smartPhotos.showItemInFolder(state.selectedItem?.path));
  elements.itemFavoriteButton.addEventListener("click", () => toggleFavorite().catch(showError));
  elements.assignRegionButton.addEventListener("click", () => assignSelectedRegion().catch(showError));
  elements.clearRegionButton.addEventListener("click", () => clearSelectedRegion().catch(showError));
  elements.assignItemButton.addEventListener("click", () => assignWholeItem().catch(showError));
  elements.clearItemButton.addEventListener("click", () => clearWholeItem().catch(showError));
  elements.albumAddButton.addEventListener("click", () => addSelectedItemToAlbum().catch(showError));
  elements.newPersonaButton.addEventListener("click", () => createPersona().catch(showError));
  elements.unknownAssignButton.addEventListener("click", () => assignUnknownClusters().catch(showError));
  elements.newAlbumButton.addEventListener("click", () => createAlbum().catch(showError));
  elements.deleteAlbumButton.addEventListener("click", () => deleteSelectedAlbum().catch(showError));
  elements.downloadModelsButton.addEventListener("click", () => triggerModelDownload().catch(showError));
}

async function bootstrap() {
  bindEvents();
  await loadStatus();
  await loadPersonas();
  await loadAlbums();
  await loadLibrary();
  syncPersonaSelects();
  populateAlbumSelect();
  startStatusPolling();
}

bootstrap().catch(showError);
