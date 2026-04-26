const invoke = window.__TAURI__?.core?.invoke;

if (!invoke) {
  document.body.innerHTML = "<main class='empty-state'>Tauri bridge is unavailable.</main>";
  throw new Error("Tauri bridge is unavailable");
}

const apiBaseUrl = await invoke("api_base_url");

const state = {
  view: "library",
  status: null,
  items: [],
  selectedItem: null,
  selectedDetectionId: "",
  personas: [],
  selectedPersonaId: "",
  clusters: [],
  reviewClusterIndex: 0,
  reviewSuggestions: [],
  reviewSuggestionIndex: 0,
  albums: [],
  selectedAlbumId: "",
  memories: [],
  selectedMemoryId: "",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const el = {
  nav: $$(".nav-item"),
  views: $$(".view"),
  viewEyebrow: $("#view-eyebrow"),
  viewTitle: $("#view-title"),
  viewCopy: $("#view-copy"),
  statItems: $("#stat-items"),
  statPersonas: $("#stat-personas"),
  statAlbums: $("#stat-albums"),
  statusDot: $("#status-dot"),
  statusText: $("#status-text"),
  statusProgress: $("#status-progress"),
  syncButton: $("#sync-button"),
  openRootButton: $("#open-root-button"),
  librarySearch: $("#library-search"),
  libraryType: $("#library-type"),
  libraryKind: $("#library-kind"),
  libraryPersona: $("#library-persona"),
  libraryFavorites: $("#library-favorites"),
  libraryGrid: $("#library-grid"),
  previewStage: $("#preview-stage"),
  favoriteButton: $("#favorite-button"),
  openFileButton: $("#open-file-button"),
  showFolderButton: $("#show-folder-button"),
  selectedPersonas: $("#selected-personas"),
  itemDetails: $("#item-details"),
  detectionList: $("#detection-list"),
  assignKind: $("#assign-kind"),
  assignPersona: $("#assign-persona"),
  assignNewName: $("#assign-new-name"),
  assignRegionButton: $("#assign-region-button"),
  clearRegionButton: $("#clear-region-button"),
  assignItemButton: $("#assign-item-button"),
  clearItemButton: $("#clear-item-button"),
  peopleKind: $("#people-kind"),
  newPersonaName: $("#new-persona-name"),
  newPersonaKind: $("#new-persona-kind"),
  newPersonaButton: $("#new-persona-button"),
  personaCards: $("#persona-cards"),
  personaTitle: $("#persona-title"),
  personaReferences: $("#persona-references"),
  personaItems: $("#persona-items"),
  reviewKind: $("#review-kind"),
  reloadReviewButton: $("#reload-review-button"),
  matchCard: $("#match-card"),
  reviewItems: $("#review-items"),
  newAlbumName: $("#new-album-name"),
  newAlbumButton: $("#new-album-button"),
  deleteAlbumButton: $("#delete-album-button"),
  albumList: $("#album-list"),
  albumItems: $("#album-items"),
  memoryList: $("#memory-list"),
  memorySummary: $("#memory-summary"),
  memoryItems: $("#memory-items"),
  modelList: $("#model-list"),
  downloadModelsButton: $("#download-models-button"),
};

const copy = {
  library: ["Library", "Your photos, organized locally.", "Search, browse, favorite, and make lightweight corrections."],
  people: ["People & Pets", "Named faces and companions.", "Review personas without the diagnostic clutter."],
  review: ["Review Matches", "One possible match at a time.", "Confirm or reject unknown clusters with a low-friction yes/no flow."],
  albums: ["Albums", "Small intentional collections.", "Create and browse personal albums."],
  memories: ["Memories", "Auto-built moments.", "Browse generated groups by time, place, and subjects."],
  models: ["AI Models", "Local model readiness.", "Download or verify models without exposing your media."],
};

async function fetchJson(path, options = {}) {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function fileUrl(path) {
  return path ? `${apiBaseUrl}/api/file?path=${encodeURIComponent(path)}` : "";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function debounce(fn, delay = 240) {
  let timer = 0;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

function setEmpty(node, message) {
  node.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
}

function personaName(id) {
  return state.personas.find((persona) => persona.id === id)?.name || "Unknown";
}

function personaOptions(kind = "all", includeBlank = true) {
  const personas = state.personas.filter((persona) => kind === "all" || persona.kind === kind);
  return `${includeBlank ? "<option value=''>Choose persona</option>" : ""}${personas
    .map((persona) => `<option value="${escapeHtml(persona.id)}">${escapeHtml(persona.name)}</option>`)
    .join("")}`;
}

function renderPhotoGrid(container, items, selectedId, onSelect) {
  if (!items.length) {
    setEmpty(container, "Nothing to show yet.");
    return;
  }
  container.innerHTML = "";
  for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `photo-tile${item.id === selectedId ? " selected" : ""}`;
    const thumb = fileUrl(item.thumbnailPath);
    button.innerHTML = `
      ${thumb ? `<img src="${thumb}" alt="" loading="lazy" />` : ""}
      ${item.favorite ? "<span class='tile-badge'>Favorite</span>" : ""}
    `;
    button.addEventListener("click", () => onSelect(item.id));
    container.appendChild(button);
  }
}

async function refreshStatus() {
  state.status = await fetchJson("/api/status");
  el.statItems.textContent = state.status.counts.items;
  el.statPersonas.textContent = state.status.counts.personas;
  el.statAlbums.textContent = state.status.counts.albums;
  renderStatus();
  renderModels();
}

function renderStatus() {
  const job = state.status?.job;
  if (!job) {
    el.statusText.textContent = "Ready";
    el.statusProgress.style.width = "0%";
    return;
  }
  const detail = job.detail ? `: ${job.detail}` : "";
  el.statusText.textContent = `${job.message || job.status}${detail}`;
  const percent = job.total > 0 ? Math.max(0, Math.min(100, (job.current / job.total) * 100)) : 18;
  el.statusProgress.style.width = `${percent}%`;
  el.statusDot.style.background = job.status === "failed" ? "#a63d2f" : "var(--accent)";
}

async function refreshPersonas() {
  const payload = await fetchJson(`/api/personas?kind=${encodeURIComponent(el.peopleKind.value || "all")}`);
  state.personas = payload.personas || [];
  el.libraryPersona.innerHTML = personaOptions(el.libraryKind.value, true);
  el.assignPersona.innerHTML = personaOptions(el.assignKind.value, true);
  renderPersonas();
}

async function refreshLibrary() {
  const params = new URLSearchParams({
    query: el.librarySearch.value,
    type: el.libraryType.value,
    personaKind: el.libraryKind.value,
    personaId: el.libraryPersona.value,
    favorites: el.libraryFavorites.checked ? "1" : "0",
    limit: "420",
  });
  const payload = await fetchJson(`/api/items?${params}`);
  state.items = payload.items || [];
  renderPhotoGrid(el.libraryGrid, state.items, state.selectedItem?.id || "", selectItem);
  if (!state.selectedItem && state.items[0]) {
    await selectItem(state.items[0].id);
  }
}

async function selectItem(itemId) {
  state.selectedDetectionId = "";
  state.selectedItem = await fetchJson(`/api/items/${encodeURIComponent(itemId)}`);
  renderLibrarySelection();
}

function renderLibrarySelection() {
  const item = state.selectedItem;
  renderPhotoGrid(el.libraryGrid, state.items, item?.id || "", selectItem);
  if (!item) {
    el.previewStage.textContent = "Select a photo or video";
    return;
  }

  const original = fileUrl(item.path);
  const thumb = fileUrl(item.thumbnailPath);
  const isVideo = item.mediaKind === "video" || item.mediaKind === "live_photo";
  el.previewStage.innerHTML = isVideo
    ? `<video src="${original}" controls poster="${thumb}"></video>`
    : `<img src="${original}" alt="" onerror="this.src='${thumb}'" />`;
  el.selectedPersonas.innerHTML = (item.personas || [])
    .map((persona) => `<span class="pill">${escapeHtml(persona.name)}</span>`)
    .join("");
  el.itemDetails.textContent = item.details || "";
  renderDetections();
}

function renderDetections() {
  const detections = state.selectedItem?.detections || [];
  if (!detections.length) {
    setEmpty(el.detectionList, "No face or pet regions detected here.");
    return;
  }
  el.detectionList.innerHTML = "";
  for (const detection of detections) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `detection-row${state.selectedDetectionId === detection.id ? " selected" : ""}`;
    row.innerHTML = `
      <strong>${escapeHtml(detection.label)}</strong>
      <div class="muted">${escapeHtml(detection.kind)} · ${Number(detection.confidence || 0).toFixed(2)} · ${
        detection.personaId ? escapeHtml(personaName(detection.personaId)) : "Unassigned"
      }</div>
    `;
    row.addEventListener("click", () => {
      state.selectedDetectionId = detection.id;
      el.assignKind.value = detection.kind === "face" ? "person" : "pet";
      el.assignPersona.innerHTML = personaOptions(el.assignKind.value, true);
      renderDetections();
    });
    el.detectionList.appendChild(row);
  }
}

async function assignRegion() {
  if (!state.selectedItem || !state.selectedDetectionId) return;
  await fetchJson("/api/corrections/region/assign", {
    method: "POST",
    body: JSON.stringify({
      itemId: state.selectedItem.id,
      regionId: state.selectedDetectionId,
      kind: el.assignKind.value,
      personaId: el.assignPersona.value,
      newName: el.assignNewName.value,
    }),
  });
  el.assignNewName.value = "";
  await refreshPersonas();
  await selectItem(state.selectedItem.id);
}

async function assignItem() {
  if (!state.selectedItem) return;
  await fetchJson("/api/corrections/item/assign", {
    method: "POST",
    body: JSON.stringify({
      itemId: state.selectedItem.id,
      kind: el.assignKind.value,
      personaId: el.assignPersona.value,
      newName: el.assignNewName.value,
    }),
  });
  el.assignNewName.value = "";
  await refreshPersonas();
  await selectItem(state.selectedItem.id);
}

async function clearRegion() {
  if (!state.selectedItem || !state.selectedDetectionId) return;
  await fetchJson("/api/corrections/region/clear", {
    method: "POST",
    body: JSON.stringify({ itemId: state.selectedItem.id, regionId: state.selectedDetectionId }),
  });
  await selectItem(state.selectedItem.id);
}

async function clearItem() {
  if (!state.selectedItem) return;
  await fetchJson("/api/corrections/item/clear", {
    method: "POST",
    body: JSON.stringify({ itemId: state.selectedItem.id }),
  });
  await selectItem(state.selectedItem.id);
}

function renderPersonas() {
  if (!state.personas.length) {
    setEmpty(el.personaCards, "No people or pets yet. Assign a cluster or create one.");
    return;
  }
  el.personaCards.innerHTML = "";
  for (const persona of state.personas) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `persona-card${state.selectedPersonaId === persona.id ? " selected" : ""}`;
    const avatar = persona.avatarThumbnailPath ? `<img src="${fileUrl(persona.avatarThumbnailPath)}" alt="" />` : "";
    card.innerHTML = `
      <div class="avatar">${avatar}</div>
      <div>
        <h2>${escapeHtml(persona.name)}</h2>
        <div class="muted">${escapeHtml(persona.kind)} · ${persona.referenceImageCount || 0} references</div>
      </div>
    `;
    card.addEventListener("click", () => selectPersona(persona.id));
    el.personaCards.appendChild(card);
  }
}

async function selectPersona(personaId) {
  state.selectedPersonaId = personaId;
  const persona = await fetchJson(`/api/personas/${encodeURIComponent(personaId)}`);
  el.personaTitle.textContent = persona.name;
  el.personaReferences.innerHTML = (persona.referenceImages || [])
    .map((ref) => `<img src="${fileUrl(ref.path)}" alt="" />`)
    .join("");
  renderPhotoGrid(el.personaItems, persona.items || [], "", selectItemFromPersona);
  renderPersonas();
}

async function selectItemFromPersona(itemId) {
  setView("library");
  await selectItem(itemId);
}

async function refreshReview() {
  const payload = await fetchJson(`/api/unknown-clusters?kind=${encodeURIComponent(el.reviewKind.value)}`);
  state.clusters = payload.clusters || [];
  state.reviewClusterIndex = 0;
  state.reviewSuggestionIndex = 0;
  await loadReviewSuggestions();
  await renderReview();
}

async function loadReviewSuggestions() {
  const cluster = state.clusters[state.reviewClusterIndex];
  state.reviewSuggestions = [];
  state.reviewSuggestionIndex = 0;
  if (!cluster) return;
  const payload = await fetchJson(`/api/unknown-clusters/${encodeURIComponent(cluster.id)}/suggestions?limit=8`);
  state.reviewSuggestions = payload.suggestions || [];
}

async function renderReview() {
  const cluster = state.clusters[state.reviewClusterIndex];
  if (!cluster) {
    el.matchCard.innerHTML = "<div class='empty-state'>No unknown clusters need review right now.</div>";
    setEmpty(el.reviewItems, "Confirmed clusters disappear from this queue.");
    return;
  }

  const suggestion = state.reviewSuggestions[state.reviewSuggestionIndex];
  const preview = fileUrl(cluster.previewPath);
  el.matchCard.innerHTML = `
    <div class="tiny-label">${state.reviewClusterIndex + 1} of ${state.clusters.length}</div>
    <div class="match-preview">${preview ? `<img src="${preview}" alt="" />` : ""}</div>
    <div>
      <h1>${suggestion ? `Is this ${escapeHtml(suggestion.persona.name)}?` : "Who is this?"}</h1>
      <p class="muted">${cluster.memberCount} detections across ${cluster.itemCount} items · ${escapeHtml(cluster.kind)}</p>
    </div>
    ${
      suggestion
        ? `<div class="candidate-card">
            <div class="avatar">${suggestion.persona.avatarThumbnailPath ? `<img src="${fileUrl(suggestion.persona.avatarThumbnailPath)}" alt="" />` : ""}</div>
            <div>
              <h2>${escapeHtml(suggestion.persona.name)}</h2>
              <div class="muted">${escapeHtml(suggestion.method)} score ${Number(suggestion.score).toFixed(2)}</div>
            </div>
          </div>`
        : `<div class="form-grid">
            <select id="manual-review-persona" class="input">${personaOptions(cluster.kind, true)}</select>
            <input id="manual-review-name" class="input" placeholder="Or create a new ${escapeHtml(cluster.kind)}" />
          </div>`
    }
    <div class="big-actions">
      ${
        suggestion
          ? `<button id="review-yes" class="solid-button">Yes</button><button id="review-no" class="ghost-button">No</button>`
          : `<button id="review-manual" class="solid-button">Assign</button><button id="review-skip" class="ghost-button">Skip</button>`
      }
    </div>
    ${suggestion ? `<button id="review-skip" class="ghost-button">Skip this cluster</button>` : ""}
  `;

  $("#review-yes")?.addEventListener("click", () => assignCurrentCluster(suggestion.persona.id, ""));
  $("#review-no")?.addEventListener("click", nextSuggestion);
  $("#review-manual")?.addEventListener("click", () => {
    assignCurrentCluster($("#manual-review-persona")?.value || "", $("#manual-review-name")?.value || "");
  });
  $("#review-skip")?.addEventListener("click", nextCluster);

  const items = await fetchJson("/api/unknown-clusters/items", {
    method: "POST",
    body: JSON.stringify({ clusterIds: [cluster.id] }),
  });
  renderPhotoGrid(el.reviewItems, items.items || [], "", selectItemFromPersona);
}

async function assignCurrentCluster(personaId, newName) {
  const cluster = state.clusters[state.reviewClusterIndex];
  if (!cluster) return;
  await fetchJson("/api/unknown-clusters/assign", {
    method: "POST",
    body: JSON.stringify({
      clusterIds: [cluster.id],
      personaId,
      newName,
      kind: cluster.kind,
    }),
  });
  await refreshPersonas();
  state.clusters.splice(state.reviewClusterIndex, 1);
  if (state.reviewClusterIndex >= state.clusters.length) {
    state.reviewClusterIndex = 0;
  }
  await loadReviewSuggestions();
  await renderReview();
}

async function nextSuggestion() {
  state.reviewSuggestionIndex += 1;
  if (state.reviewSuggestionIndex >= state.reviewSuggestions.length) {
    state.reviewSuggestions = [];
    state.reviewSuggestionIndex = 0;
  }
  await renderReview();
}

async function nextCluster() {
  if (!state.clusters.length) return;
  state.reviewClusterIndex = (state.reviewClusterIndex + 1) % state.clusters.length;
  await loadReviewSuggestions();
  await renderReview();
}

async function refreshAlbums() {
  const payload = await fetchJson("/api/albums");
  state.albums = payload.albums || [];
  renderCollectionList(el.albumList, state.albums, state.selectedAlbumId, selectAlbum);
}

async function selectAlbum(albumId) {
  state.selectedAlbumId = albumId;
  const album = await fetchJson(`/api/albums/${encodeURIComponent(albumId)}`);
  renderCollectionList(el.albumList, state.albums, state.selectedAlbumId, selectAlbum);
  renderPhotoGrid(el.albumItems, album.items || [], "", selectItemFromPersona);
}

async function refreshMemories() {
  const payload = await fetchJson("/api/memories");
  state.memories = payload.memories || [];
  renderCollectionList(el.memoryList, state.memories, state.selectedMemoryId, selectMemory, "No memories yet.");
}

async function selectMemory(memoryId) {
  state.selectedMemoryId = memoryId;
  const memory = await fetchJson(`/api/memories/${encodeURIComponent(memoryId)}`);
  renderCollectionList(el.memoryList, state.memories, state.selectedMemoryId, selectMemory);
  el.memorySummary.textContent = memory.summary || memory.subtitle || "Memory";
  renderPhotoGrid(el.memoryItems, memory.items || [], "", selectItemFromPersona);
}

function renderCollectionList(container, entries, selectedId, onSelect, empty = "Nothing here yet.") {
  if (!entries.length) {
    setEmpty(container, empty);
    return;
  }
  container.innerHTML = "";
  for (const entry of entries) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `collection-item${entry.id === selectedId ? " selected" : ""}`;
    button.innerHTML = `<strong>${escapeHtml(entry.name || entry.title)}</strong><div class="muted">${entry.itemCount || 0} items</div>`;
    button.addEventListener("click", () => onSelect(entry.id));
    container.appendChild(button);
  }
}

function renderModels() {
  const models = state.status?.models || [];
  if (!models.length) {
    setEmpty(el.modelList, "No model metadata available.");
    return;
  }
  el.modelList.innerHTML = models
    .map(
      (model) => `
        <div class="model-card">
          <div>
            <strong>${escapeHtml(model.title)}</strong>
            <div class="muted">${escapeHtml(model.description || model.id)}</div>
          </div>
          <span class="pill">${model.installed ? "Installed" : "Missing"}</span>
        </div>
      `,
    )
    .join("");
}

function setView(view) {
  state.view = view;
  for (const button of el.nav) {
    button.classList.toggle("active", button.dataset.view === view);
  }
  for (const section of el.views) {
    section.classList.toggle("active", section.dataset.view === view);
  }
  const [eyebrow, title, viewCopy] = copy[view];
  el.viewEyebrow.textContent = eyebrow;
  el.viewTitle.textContent = title;
  el.viewCopy.textContent = viewCopy;
}

async function refreshAll() {
  await refreshStatus();
  await refreshPersonas();
  await refreshLibrary();
  await refreshReview();
  await refreshAlbums();
  await refreshMemories();
}

function wireEvents() {
  for (const button of el.nav) {
    button.addEventListener("click", () => setView(button.dataset.view));
  }
  el.syncButton.addEventListener("click", async () => {
    await fetchJson("/api/jobs/sync", { method: "POST", body: "{}" });
    await refreshStatus();
  });
  el.openRootButton.addEventListener("click", () => invoke("open_path", { path: state.status?.mediaRoot || "" }));
  const refreshLibraryDebounced = debounce(refreshLibrary);
  for (const node of [el.librarySearch, el.libraryType, el.libraryKind, el.libraryPersona, el.libraryFavorites]) {
    node.addEventListener("input", refreshLibraryDebounced);
    node.addEventListener("change", refreshLibraryDebounced);
  }
  el.libraryKind.addEventListener("change", () => {
    el.libraryPersona.innerHTML = personaOptions(el.libraryKind.value, true);
  });
  el.assignKind.addEventListener("change", () => {
    el.assignPersona.innerHTML = personaOptions(el.assignKind.value, true);
  });
  el.favoriteButton.addEventListener("click", async () => {
    if (!state.selectedItem) return;
    await fetchJson(`/api/items/${encodeURIComponent(state.selectedItem.id)}/toggle-favorite`, { method: "POST", body: "{}" });
    await selectItem(state.selectedItem.id);
    await refreshLibrary();
  });
  el.openFileButton.addEventListener("click", () => invoke("open_path", { path: state.selectedItem?.path || "" }));
  el.showFolderButton.addEventListener("click", () => invoke("show_item_in_folder", { path: state.selectedItem?.path || "" }));
  el.assignRegionButton.addEventListener("click", assignRegion);
  el.clearRegionButton.addEventListener("click", clearRegion);
  el.assignItemButton.addEventListener("click", assignItem);
  el.clearItemButton.addEventListener("click", clearItem);
  el.peopleKind.addEventListener("change", refreshPersonas);
  el.newPersonaButton.addEventListener("click", async () => {
    if (!el.newPersonaName.value.trim()) return;
    await fetchJson("/api/personas", {
      method: "POST",
      body: JSON.stringify({ name: el.newPersonaName.value, kind: el.newPersonaKind.value }),
    });
    el.newPersonaName.value = "";
    await refreshPersonas();
  });
  el.reviewKind.addEventListener("change", refreshReview);
  el.reloadReviewButton.addEventListener("click", refreshReview);
  el.newAlbumButton.addEventListener("click", async () => {
    if (!el.newAlbumName.value.trim()) return;
    await fetchJson("/api/albums", { method: "POST", body: JSON.stringify({ name: el.newAlbumName.value, itemIds: [] }) });
    el.newAlbumName.value = "";
    await refreshAlbums();
  });
  el.deleteAlbumButton.addEventListener("click", async () => {
    if (!state.selectedAlbumId) return;
    await fetchJson(`/api/albums/${encodeURIComponent(state.selectedAlbumId)}`, { method: "DELETE" });
    state.selectedAlbumId = "";
    await refreshAlbums();
    setEmpty(el.albumItems, "Select an album.");
  });
  el.downloadModelsButton.addEventListener("click", async () => {
    await fetchJson("/api/jobs/models/recommended", { method: "POST", body: "{}" });
    await refreshStatus();
  });
}

wireEvents();
await refreshAll();
setInterval(async () => {
  try {
    await refreshStatus();
  } catch (error) {
    el.statusText.textContent = String(error);
  }
}, 3500);
