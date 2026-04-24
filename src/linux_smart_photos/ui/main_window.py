from __future__ import annotations

from time import monotonic
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, QSize, Qt, Signal
from PySide6.QtGui import QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig
from ..services.library import LibraryService, ProgressUpdate, UnknownPersonaCluster
from .dialogs import AlbumDialog, AssignPersonaDialog, CorrectionsDialog
from .widgets import MediaGridWidget


PAGE_ITEM_PREVIEW_LIMIT = 720
CLUSTER_RENDER_CHUNK_SIZE = 128
LIVE_REFRESH_INTERVAL_MS = 600


class TaskStatusRow(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title_label = QLabel(title)
        self.title_label.setMinimumWidth(128)
        self.message_label = QLabel("Idle")
        self.message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(240)
        self.progress_bar.setMaximumWidth(320)
        self.progress_bar.setTextVisible(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.message_label, 1)
        layout.addWidget(self.progress_bar)

        self.set_idle("Idle")

    def set_idle(self, message: str = "Idle") -> None:
        self.message_label.setText(message)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("idle")

    def set_active(
        self,
        message: str,
        *,
        current: int = 0,
        total: int = 0,
        indeterminate: bool = True,
    ) -> None:
        self.message_label.setText(message)
        if indeterminate or total <= 0:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("working")
            return
        bounded_current = max(0, min(current, total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(bounded_current)
        self.progress_bar.setFormat(f"{bounded_current}/{total}")

    def set_complete(self, message: str) -> None:
        self.message_label.setText(message)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_bar.setFormat("done")

    def set_error(self, message: str) -> None:
        self.message_label.setText(message)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("error")


class BackgroundTaskWorker(QObject):
    progress = Signal(object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, config: AppConfig, task_name: str, model_id: str = "") -> None:
        super().__init__()
        self.config = config
        self.task_name = task_name
        self.model_id = model_id

    def run(self) -> None:
        service = LibraryService(self.config)
        try:
            if self.task_name == "startup":
                missing_model_ids = service.missing_recommended_model_ids()
                download_error = ""
                if missing_model_ids:
                    try:
                        service.download_recommended_models(progress_callback=self._emit_progress)
                    except Exception as exc:
                        download_error = str(exc)
                summary = service.sync(progress_callback=self._emit_progress)
                self.completed.emit(
                    {
                        "task": self.task_name,
                        "summary": summary,
                        "missing_model_ids": missing_model_ids,
                        "download_error": download_error,
                    }
                )
                return

            if self.task_name == "sync":
                summary = service.sync(progress_callback=self._emit_progress)
                self.completed.emit({"task": self.task_name, "summary": summary})
                return

            if self.task_name == "download_recommended_models":
                missing_model_ids = service.missing_recommended_model_ids()
                paths = service.download_recommended_models(progress_callback=self._emit_progress)
                self.completed.emit(
                    {
                        "task": self.task_name,
                        "paths": paths,
                        "missing_model_ids": missing_model_ids,
                    }
                )
                return

            if self.task_name == "download_model":
                path = service.download_model(self.model_id, progress_callback=self._emit_progress)
                self.completed.emit({"task": self.task_name, "path": path, "model_id": self.model_id})
                return

            raise ValueError(f"Unsupported background task: {self.task_name}")
        except Exception as exc:
            self.failed.emit(str(exc))

    def _emit_progress(self, update: ProgressUpdate) -> None:
        self.progress.emit(update)


class UnknownClustersWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        config: AppConfig,
        kind: str,
        *,
        allow_stale_cache: bool = False,
        build_if_missing: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.kind = kind
        self.allow_stale_cache = allow_stale_cache
        self.build_if_missing = build_if_missing

    def run(self) -> None:
        try:
            service = LibraryService(self.config)
            clusters = service.list_unknown_persona_clusters(
                kind=self.kind,
                allow_stale_cache=self.allow_stale_cache,
                build_if_missing=self.build_if_missing,
            )
            self.completed.emit(clusters)
        except Exception as exc:
            self.failed.emit(str(exc))


class UnknownClusterCacheWorker(QObject):
    progress = Signal(object)
    completed = Signal(bool)
    failed = Signal(str)

    def __init__(self, config: AppConfig, *, partial: bool) -> None:
        super().__init__()
        self.config = config
        self.partial = partial

    def run(self) -> None:
        try:
            service = LibraryService(self.config)
            service.rebuild_unknown_cluster_caches(
                partial=self.partial,
                progress_callback=self._emit_progress,
            )
            self.completed.emit(self.partial)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _emit_progress(self, update: ProgressUpdate) -> None:
        self.progress.emit(update)


class ItemLoadWorker(QObject):
    completed = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, config: AppConfig, request_id: int, mode: str, payload: Any) -> None:
        super().__init__()
        self.config = config
        self.request_id = request_id
        self.mode = mode
        self.payload = payload

    def run(self) -> None:
        try:
            service = LibraryService(self.config)
            if self.mode == "persona":
                items = service.items_for_persona(str(self.payload), limit=PAGE_ITEM_PREVIEW_LIMIT)
            elif self.mode == "unknown_clusters":
                items = service.items_for_unknown_clusters(self.payload, limit=PAGE_ITEM_PREVIEW_LIMIT)
            elif self.mode == "album":
                items = service.items_for_album(str(self.payload), limit=PAGE_ITEM_PREVIEW_LIMIT)
            elif self.mode == "memory":
                items = service.items_for_memory(str(self.payload), limit=PAGE_ITEM_PREVIEW_LIMIT)
            else:
                raise ValueError(f"Unsupported item load mode: {self.mode}")
            self.completed.emit(self.request_id, items)
        except Exception as exc:
            self.failed.emit(self.request_id, str(exc))


class LibraryPage(QWidget):
    def __init__(self, service: LibraryService, owner: "MainWindow") -> None:
        super().__init__()
        self.service = service
        self.owner = owner

        controls = QHBoxLayout()

        self.search_box = self._build_combo_line_placeholder()
        self.search_box.textChanged.connect(self.refresh)

        self.type_filter = QComboBox()
        self.type_filter.addItem("All Media", "all")
        self.type_filter.addItem("Images", "image")
        self.type_filter.addItem("Videos", "video")
        self.type_filter.addItem("GIFs", "gif")
        self.type_filter.addItem("Live Photos", "live_photo")
        self.type_filter.currentIndexChanged.connect(self.refresh)

        self.kind_filter = QComboBox()
        self.kind_filter.addItem("All Personas", "all")
        self.kind_filter.addItem("People", "person")
        self.kind_filter.addItem("Pets", "pet")
        self.kind_filter.currentIndexChanged.connect(self._refresh_persona_filter)
        self.kind_filter.currentIndexChanged.connect(self.refresh)

        self.persona_filter = QComboBox()
        self.persona_filter.addItem("Any Persona", "")
        self.persona_filter.currentIndexChanged.connect(self.refresh)

        self.favorites_only = QCheckBox("Favorites")
        self.favorites_only.stateChanged.connect(self.refresh)

        self.refresh_button = QPushButton("Refresh Library")
        self.refresh_button.clicked.connect(self.owner.sync_library)
        self.correct_button = QPushButton("Assign / Correct")
        self.correct_button.clicked.connect(self.open_corrections)
        self.album_button = QPushButton("Add To Album")
        self.album_button.clicked.connect(self.add_to_album)
        self.favorite_button = QPushButton("Toggle Favorite")
        self.favorite_button.clicked.connect(self.toggle_favorite)

        controls.addWidget(self.search_box)
        controls.addWidget(self.type_filter)
        controls.addWidget(self.kind_filter)
        controls.addWidget(self.persona_filter)
        controls.addWidget(self.favorites_only)
        controls.addStretch(1)
        controls.addWidget(self.correct_button)
        controls.addWidget(self.album_button)
        controls.addWidget(self.favorite_button)
        controls.addWidget(self.refresh_button)

        self.grid = MediaGridWidget(service)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.grid)

        self._refresh_persona_filter()

    def _build_combo_line_placeholder(self):
        from PySide6.QtWidgets import QLineEdit

        line_edit = QLineEdit()
        line_edit.setPlaceholderText(
            "Search photos and videos. Examples: cat, birthday, person:alice type:video"
        )
        return line_edit

    def _refresh_persona_filter(self, *_args) -> None:
        selected = self.persona_filter.currentData()
        self.persona_filter.blockSignals(True)
        self.persona_filter.clear()
        self.persona_filter.addItem("Any Persona", "")
        for persona in self.service.list_personas(kind=str(self.kind_filter.currentData())):
            self.persona_filter.addItem(persona.name, persona.id)
        index = self.persona_filter.findData(selected)
        self.persona_filter.setCurrentIndex(max(index, 0))
        self.persona_filter.blockSignals(False)

    def refresh(self, *_args, limit: int | None = None) -> None:
        self.grid.set_page_loader(
            lambda offset, page_limit: self.service.search_items_page(
                query=self.search_box.text(),
                media_kind=str(self.type_filter.currentData()),
                persona_kind=str(self.kind_filter.currentData()),
                persona_id=str(self.persona_filter.currentData()),
                favorites_only=self.favorites_only.isChecked(),
                offset=offset,
                limit=page_limit,
            ),
            empty_message="No media items match the current view",
        )

    def open_corrections(self, *_args) -> None:
        item_id = self.grid.current_item_id()
        if not item_id:
            QMessageBox.warning(self, "No Item Selected", "Choose a photo or video first.")
            return
        dialog = CorrectionsDialog(self.service, item_id, self)
        dialog.exec()
        self.owner.refresh_views()

    def add_to_album(self, *_args) -> None:
        item_ids = self.grid.selected_item_ids() or ([self.grid.current_item_id()] if self.grid.current_item_id() else [])
        if not item_ids:
            QMessageBox.warning(self, "No Items Selected", "Select at least one media item first.")
            return
        dialog = AlbumDialog(self.service.list_albums(), self)
        if dialog.exec() != QDialog.Accepted:
            return
        choice = dialog.selection()
        if choice["album_id"]:
            self.service.add_items_to_album(choice["album_id"], item_ids)
        else:
            self.service.create_album(choice["new_name"], item_ids)
        self.owner.refresh_views()

    def toggle_favorite(self, *_args) -> None:
        item_ids = self.grid.selected_item_ids() or ([self.grid.current_item_id()] if self.grid.current_item_id() else [])
        if not item_ids:
            QMessageBox.warning(self, "No Items Selected", "Choose one or more items first.")
            return
        self.service.toggle_favorite(item_ids)
        self.owner.refresh_views()

    def set_busy(self, busy: bool) -> None:
        for button in (
            self.refresh_button,
            self.correct_button,
            self.album_button,
            self.favorite_button,
        ):
            button.setEnabled(not busy)


class PeoplePage(QWidget):
    def __init__(self, service: LibraryService, owner: "MainWindow") -> None:
        super().__init__()
        self.service = service
        self.owner = owner
        self._items_thread: QThread | None = None
        self._items_worker: ItemLoadWorker | None = None
        self._pending_item_request: tuple[int, str] | None = None
        self._latest_item_request_id = 0

        self.kind_filter = QComboBox()
        self.kind_filter.addItem("All Personas", "all")
        self.kind_filter.addItem("People", "person")
        self.kind_filter.addItem("Pets", "pet")
        self.kind_filter.currentIndexChanged.connect(self.refresh)

        self.persona_list = QListWidget()
        self.persona_list.currentItemChanged.connect(self._show_persona_items)

        self.new_persona_button = QPushButton("New Persona")
        self.new_persona_button.clicked.connect(self._create_persona)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.kind_filter)
        left_layout.addWidget(self.persona_list)
        left_layout.addWidget(self.new_persona_button)

        left_panel = QWidget()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(320)

        self.grid = MediaGridWidget(service)
        self.correct_button = QPushButton("Correct Current Selection")
        self.correct_button.clicked.connect(self._open_corrections)
        self.reference_summary = QLabel("Reference crops will appear here.")
        self.reference_list = QListWidget()
        self.reference_list.setViewMode(QListView.IconMode)
        self.reference_list.setFlow(QListView.LeftToRight)
        self.reference_list.setWrapping(False)
        self.reference_list.setResizeMode(QListView.Adjust)
        self.reference_list.setMovement(QListView.Static)
        self.reference_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.reference_list.setIconSize(QSize(88, 88))
        self.reference_list.setMinimumHeight(132)
        self.reference_list.setMaximumHeight(148)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.correct_button)
        right_layout.addWidget(self.reference_summary)
        right_layout.addWidget(self.reference_list)
        right_layout.addWidget(self.grid)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)

        outer_layout = QHBoxLayout(self)
        outer_layout.addWidget(left_panel)
        outer_layout.addWidget(right_panel, 1)

    def refresh(self, *_args, load_items: bool = True) -> None:
        selected_id = self._current_persona_id()
        self.persona_list.blockSignals(True)
        self.persona_list.clear()
        for persona in self.service.list_personas(kind=str(self.kind_filter.currentData())):
            item = QListWidgetItem(f"{persona.name} ({persona.kind})")
            item.setData(Qt.UserRole, persona.id)
            self.persona_list.addItem(item)
            if persona.id == selected_id:
                self.persona_list.setCurrentItem(item)
        self.persona_list.blockSignals(False)
        if self.persona_list.count() and not self.persona_list.currentItem():
            self.persona_list.setCurrentRow(0)
        if load_items:
            self._show_persona_items()

    def _show_persona_items(self, *_args) -> None:
        persona_id = self._current_persona_id()
        self._show_reference_images(persona_id)
        if not persona_id:
            self.grid.set_items([])
            return
        self.grid.set_page_loader(
            lambda offset, page_limit, persona_id=persona_id: self.service.items_for_persona_page(
                persona_id,
                offset=offset,
                limit=page_limit,
            ),
            empty_message="No media items belong to this persona yet.",
        )

    def _show_reference_images(self, persona_id: str) -> None:
        references = self.service.persona_reference_images(persona_id) if persona_id else []
        self.reference_list.clear()
        if not references:
            self.reference_summary.setText("Reference crops: none yet.")
            return

        self.reference_summary.setText(f"Reference crops: {len(references)}")
        for reference in references:
            icon = QIcon(reference["path"]) if reference.get("path") else QIcon()
            item = QListWidgetItem(icon, reference.get("label", "reference"))
            tooltip = "\n".join(
                filter(
                    None,
                    [
                        reference.get("kind", ""),
                        reference.get("label", ""),
                        reference.get("path", ""),
                    ],
                )
            )
            if tooltip:
                item.setToolTip(tooltip)
            self.reference_list.addItem(item)

    def _create_persona(self, *_args) -> None:
        kind = str(self.kind_filter.currentData())
        if kind == "all":
            kind = "person"
        name, ok = QInputDialog.getText(self, "New Persona", f"Enter the {kind} name")
        if not ok or not name.strip():
            return
        self.service.create_persona(name, kind)
        self.owner.refresh_views()

    def _open_corrections(self, *_args) -> None:
        item_id = self.grid.current_item_id()
        if not item_id:
            return
        dialog = CorrectionsDialog(self.service, item_id, self)
        dialog.exec()
        self.owner.refresh_views()

    def _current_persona_id(self) -> str:
        current = self.persona_list.currentItem()
        if not current:
            return ""
        return str(current.data(Qt.UserRole))

    def _queue_item_load(self, persona_id: str) -> None:
        self._latest_item_request_id += 1
        self._pending_item_request = (self._latest_item_request_id, persona_id)
        if self._items_thread is None:
            self._start_next_item_request()

    def _start_next_item_request(self) -> None:
        if self._pending_item_request is None or self._items_thread is not None:
            return
        request_id, persona_id = self._pending_item_request
        self._pending_item_request = None

        thread = QThread(self)
        worker = ItemLoadWorker(self.service.config, request_id, "persona", persona_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._handle_items_loaded)
        worker.failed.connect(self._handle_items_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_item_handles)

        self._items_thread = thread
        self._items_worker = worker
        thread.start()

    def _handle_items_loaded(self, request_id: int, payload: Any) -> None:
        if request_id != self._latest_item_request_id:
            return
        items = payload if isinstance(payload, list) else []
        self.grid.set_items(items)

    def _handle_items_failed(self, request_id: int, message: str) -> None:
        if request_id != self._latest_item_request_id:
            return
        QMessageBox.warning(self, "Persona Items", f"Unable to load persona items: {message}")

    def _clear_item_handles(self) -> None:
        self._items_thread = None
        self._items_worker = None
        if self._pending_item_request is not None:
            QTimer.singleShot(0, self._start_next_item_request)

    def set_busy(self, busy: bool) -> None:
        self.new_persona_button.setEnabled(not busy)
        self.correct_button.setEnabled(not busy)


class UnknownClustersPage(QWidget):
    def __init__(self, service: LibraryService, owner: "MainWindow") -> None:
        super().__init__()
        self.service = service
        self.owner = owner
        self._clusters_by_id: dict[str, UnknownPersonaCluster] = {}
        self._refresh_thread: QThread | None = None
        self._refresh_worker: UnknownClustersWorker | None = None
        self._refresh_pending = False
        self._refresh_kind = "all"
        self._pending_render_clusters: list[UnknownPersonaCluster] = []
        self._pending_selected_cluster_ids: set[str] = set()
        self._render_cursor = 0
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_next_cluster_chunk)
        self._items_thread: QThread | None = None
        self._items_worker: ItemLoadWorker | None = None
        self._pending_item_request: tuple[int, list[UnknownPersonaCluster], str, int] | None = None
        self._latest_item_request_id = 0

        self.kind_filter = QComboBox()
        self.kind_filter.addItem("All Unknown", "all")
        self.kind_filter.addItem("People", "person")
        self.kind_filter.addItem("Pets", "pet")
        self.kind_filter.currentIndexChanged.connect(self.refresh)

        self.cluster_list = QListWidget()
        self.cluster_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.cluster_list.setIconSize(QSize(72, 72))
        self.cluster_list.itemSelectionChanged.connect(self._show_selected_clusters)

        self.refresh_button = QPushButton("Refresh Clusters")
        self.refresh_button.clicked.connect(self.refresh)
        self.assign_button = QPushButton("Assign Selected Clusters")
        self.assign_button.clicked.connect(self._assign_selected_clusters)
        self.review_button = QPushButton("Review Current Item")
        self.review_button.clicked.connect(self._open_corrections)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.kind_filter)
        left_layout.addWidget(self.cluster_list)
        left_layout.addWidget(self.refresh_button)
        left_layout.addWidget(self.assign_button)
        left_layout.addWidget(self.review_button)

        left_panel = QWidget()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(380)

        self.summary = QLabel("Select one or more unknown clusters to review or assign.")
        self.summary.setWordWrap(True)
        self.grid = MediaGridWidget(service)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.summary)
        right_layout.addWidget(self.grid)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)

        outer_layout = QHBoxLayout(self)
        outer_layout.addWidget(left_panel)
        outer_layout.addWidget(right_panel, 1)

    def refresh(self, *_args) -> None:
        self._refresh_kind = str(self.kind_filter.currentData())
        if self._refresh_thread is not None:
            self._refresh_pending = True
            return

        is_background_analysis_active = self.owner._active_task in {"startup", "sync"}

        if self.cluster_list.count():
            self.summary.setText("Refreshing unknown clusters in the background...")
            self.owner.set_cluster_view_status_loading("Refreshing unknown clusters from cache/background worker")
        else:
            self.summary.setText("Loading unknown clusters...")
            self.cluster_list.setEnabled(False)
            self.owner.set_cluster_view_status_loading("Loading unknown clusters")
        self.refresh_button.setEnabled(False)

        thread = QThread(self)
        worker = UnknownClustersWorker(
            self.service.config,
            self._refresh_kind,
            allow_stale_cache=is_background_analysis_active,
            build_if_missing=not is_background_analysis_active,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._handle_refresh_completed)
        worker.failed.connect(self._handle_refresh_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_refresh_handles)

        self._refresh_thread = thread
        self._refresh_worker = worker
        thread.start()

    def _show_selected_clusters(self, *_args) -> None:
        clusters = self._selected_clusters()
        if not self._clusters_by_id:
            if self.owner._active_task in {"startup", "sync"}:
                self.summary.setText(
                    "Background analysis is still running. Unknown face and pet clusters will appear automatically."
                )
            else:
                self.summary.setText("No unknown people or pet clusters are waiting for assignment.")
            self.grid.set_items([])
            return
        if not clusters:
            self.summary.setText("Select one or more unknown clusters to review or assign.")
            self.grid.set_items([])
            return

        total_members = sum(cluster.member_count for cluster in clusters)
        total_items = len({item_id for cluster in clusters for item_id in cluster.item_ids})
        cluster_names = ", ".join(self._cluster_kind_name(cluster) for cluster in clusters[:4])
        if len(clusters) > 4:
            cluster_names = f"{cluster_names}, ..."
        self.summary.setText(
            f"Selected clusters: {len(clusters)}\n"
            f"Detections: {total_members}\n"
            f"Items: {total_items}\n"
            f"Types: {cluster_names or 'unknown'}"
        )
        self.grid.set_page_loader(
            lambda offset, page_limit, clusters=clusters: self.service.items_for_unknown_clusters_page(
                clusters,
                offset=offset,
                limit=page_limit,
            ),
            empty_message="No media items are available for the selected unknown clusters.",
        )

    def _assign_selected_clusters(self, *_args) -> None:
        clusters = self._selected_clusters()
        if not clusters:
            QMessageBox.warning(self, "No Clusters Selected", "Select one or more clusters first.")
            return

        kinds = {cluster.kind for cluster in clusters}
        if len(kinds) > 1:
            QMessageBox.warning(
                self,
                "Mixed Cluster Types",
                "Select only people clusters or only pet clusters for one assignment action.",
            )
            return

        dialog = AssignPersonaDialog(
            self.service.list_personas(),
            suggested_kind=clusters[0].kind,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        choice = dialog.selection()
        try:
            self.service.assign_unknown_clusters_to_persona(
                clusters,
                persona_id=choice["persona_id"],
                new_name=choice["new_name"],
                kind=choice["kind"],
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Assignment Failed", str(exc))
            return
        self.owner.refresh_views()

    def _open_corrections(self, *_args) -> None:
        item_id = self.grid.current_item_id()
        if not item_id:
            QMessageBox.warning(self, "No Item Selected", "Choose an item from the selected clusters first.")
            return
        dialog = CorrectionsDialog(self.service, item_id, self)
        dialog.exec()
        self.owner.refresh_views()

    def _selected_clusters(self) -> list[UnknownPersonaCluster]:
        clusters: list[UnknownPersonaCluster] = []
        for item in self.cluster_list.selectedItems():
            cluster_id = str(item.data(Qt.UserRole))
            cluster = self._clusters_by_id.get(cluster_id)
            if cluster is not None:
                clusters.append(cluster)
        return clusters

    def _cluster_title(self, cluster: UnknownPersonaCluster, index: int) -> str:
        if cluster.kind == "person":
            base = f"Unknown person {index}"
        else:
            base = f"Unknown {cluster.label or 'pet'} {index}"
        return f"{base}  •  {cluster.member_count} detections / {cluster.item_count} items"

    def _cluster_tooltip(self, cluster: UnknownPersonaCluster) -> str:
        return "\n".join(
            [
                f"Kind: {cluster.kind}",
                f"Label: {cluster.label}",
                f"Detections: {cluster.member_count}",
                f"Items: {cluster.item_count}",
                f"Average confidence: {cluster.average_confidence:.2f}",
            ]
        )

    def _cluster_kind_name(self, cluster: UnknownPersonaCluster) -> str:
        if cluster.kind == "person":
            return "person"
        return cluster.label or "pet"

    def set_busy(self, busy: bool) -> None:
        self.refresh_button.setEnabled(self._refresh_thread is None)
        self.assign_button.setEnabled(not busy)
        self.review_button.setEnabled(not busy)

    def _handle_refresh_completed(self, payload: Any) -> None:
        clusters = payload if isinstance(payload, list) else []
        selected_ids = {str(item.data(Qt.UserRole)) for item in self.cluster_list.selectedItems()}
        self._clusters_by_id = {cluster.id: cluster for cluster in clusters if isinstance(cluster, UnknownPersonaCluster)}
        self._pending_render_clusters = [
            cluster for cluster in clusters if isinstance(cluster, UnknownPersonaCluster)
        ]
        self._pending_selected_cluster_ids = selected_ids
        self._render_cursor = 0
        self.cluster_list.blockSignals(True)
        self.cluster_list.clear()
        self.cluster_list.blockSignals(False)
        if not self._pending_render_clusters and self.owner._active_task in {"startup", "sync"}:
            self.summary.setText(
                "Scanning in background. Unknown clusters will appear automatically as analyzed items accumulate."
            )
            self.owner.set_cluster_view_status_complete("No unknown clusters yet; waiting for analyzed face/pet detections")
        elif self.owner._active_task in {"startup", "sync"}:
            self.summary.setText(
                f"Rendering {len(self._pending_render_clusters)} live unknown clusters from analyzed items so far..."
            )
            self.owner.set_cluster_view_status_loading(
                f"Rendering {len(self._pending_render_clusters)} live unknown clusters"
            )
        else:
            self.summary.setText(f"Rendering {len(self._pending_render_clusters)} unknown clusters...")
            self.owner.set_cluster_view_status_loading(
                f"Rendering {len(self._pending_render_clusters)} unknown clusters"
            )
        self._render_timer.start(0)

    def _handle_refresh_failed(self, message: str) -> None:
        self.cluster_list.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.summary.setText(f"Unable to load unknown clusters: {message}")
        self.owner.set_cluster_view_status_error(f"Unknown clusters failed to load: {message}")

    def _clear_refresh_handles(self) -> None:
        self._refresh_thread = None
        self._refresh_worker = None
        if self._refresh_pending:
            self._refresh_pending = False
            QTimer.singleShot(0, self.refresh)

    def _render_next_cluster_chunk(self) -> None:
        if self._render_cursor >= len(self._pending_render_clusters):
            self.cluster_list.setEnabled(True)
            self.refresh_button.setEnabled(True)
            if not self.cluster_list.selectedItems() and self.cluster_list.count():
                self.cluster_list.item(0).setSelected(True)
                self.cluster_list.setCurrentRow(0)
            self.owner.set_cluster_view_status_complete(
                f"Unknown clusters ready: {len(self._pending_render_clusters)} cluster(s)"
            )
            self._show_selected_clusters()
            return

        batch = self._pending_render_clusters[
            self._render_cursor : self._render_cursor + CLUSTER_RENDER_CHUNK_SIZE
        ]
        self.cluster_list.blockSignals(True)
        for cluster in batch:
            index = self._render_cursor + 1
            title = self._cluster_title(cluster, index)
            icon = QIcon(cluster.preview_path) if cluster.preview_path else QIcon()
            item = QListWidgetItem(icon, title)
            item.setData(Qt.UserRole, cluster.id)
            item.setToolTip(self._cluster_tooltip(cluster))
            self.cluster_list.addItem(item)
            if cluster.id in self._pending_selected_cluster_ids:
                item.setSelected(True)
            self._render_cursor += 1
        self.cluster_list.blockSignals(False)
        self.summary.setText(
            f"Rendering {len(self._pending_render_clusters)} unknown clusters..."
            f" ({self._render_cursor}/{len(self._pending_render_clusters)})"
        )
        self._render_timer.start(0)

    def _queue_item_load(
        self,
        clusters: list[UnknownPersonaCluster],
        *,
        summary_text: str,
        total_items: int,
    ) -> None:
        self._latest_item_request_id += 1
        self._pending_item_request = (
            self._latest_item_request_id,
            clusters,
            summary_text,
            total_items,
        )
        if self._items_thread is None:
            self._start_next_item_request()

    def _start_next_item_request(self) -> None:
        if self._pending_item_request is None or self._items_thread is not None:
            return
        request_id, clusters, summary_text, total_items = self._pending_item_request
        self._pending_item_request = None
        self.summary.setText(
            f"{summary_text}\nShowing latest items while the background load completes..."
        )

        thread = QThread(self)
        worker = ItemLoadWorker(self.service.config, request_id, "unknown_clusters", clusters)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(
            lambda completed_request_id, payload: self._handle_item_load_completed(
                completed_request_id,
                payload,
                summary_text=summary_text,
                total_items=total_items,
            )
        )
        worker.failed.connect(self._handle_item_load_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_item_handles)

        self._items_thread = thread
        self._items_worker = worker
        thread.start()

    def _handle_item_load_completed(
        self,
        request_id: int,
        payload: Any,
        *,
        summary_text: str,
        total_items: int,
    ) -> None:
        if request_id != self._latest_item_request_id:
            return
        items = payload if isinstance(payload, list) else []
        self.grid.set_items(items)
        if total_items > len(items):
            self.summary.setText(
                f"{summary_text}\nShowing latest {len(items)} of {total_items} items."
            )
        else:
            self.summary.setText(summary_text)

    def _handle_item_load_failed(self, request_id: int, message: str) -> None:
        if request_id != self._latest_item_request_id:
            return
        QMessageBox.warning(self, "Unknown Clusters", f"Unable to load cluster items: {message}")

    def _clear_item_handles(self) -> None:
        self._items_thread = None
        self._items_worker = None
        if self._pending_item_request is not None:
            QTimer.singleShot(0, self._start_next_item_request)


class AlbumsPage(QWidget):
    def __init__(self, service: LibraryService, owner: "MainWindow") -> None:
        super().__init__()
        self.service = service
        self.owner = owner

        self.album_list = QListWidget()
        self.album_list.currentItemChanged.connect(self._show_album_items)

        self.new_album_button = QPushButton("New Empty Album")
        self.new_album_button.clicked.connect(self._new_album)
        self.delete_album_button = QPushButton("Delete Album")
        self.delete_album_button.clicked.connect(self._delete_album)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.album_list)
        left_layout.addWidget(self.new_album_button)
        left_layout.addWidget(self.delete_album_button)

        left_panel = QWidget()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(320)

        self.grid = MediaGridWidget(service)

        outer_layout = QHBoxLayout(self)
        outer_layout.addWidget(left_panel)
        outer_layout.addWidget(self.grid, 1)

    def refresh(self, *_args, load_items: bool = True) -> None:
        selected_id = self._current_album_id()
        self.album_list.blockSignals(True)
        self.album_list.clear()
        for album in self.service.list_albums():
            item = QListWidgetItem(f"{album.name} ({len(album.item_ids)})")
            item.setData(Qt.UserRole, album.id)
            self.album_list.addItem(item)
            if album.id == selected_id:
                self.album_list.setCurrentItem(item)
        self.album_list.blockSignals(False)
        if self.album_list.count() and not self.album_list.currentItem():
            self.album_list.setCurrentRow(0)
        if load_items:
            self._show_album_items()

    def _show_album_items(self, *_args) -> None:
        album_id = self._current_album_id()
        if not album_id:
            self.grid.set_items([])
            return
        self.grid.set_page_loader(
            lambda offset, page_limit, album_id=album_id: self.service.items_for_album_page(
                album_id,
                offset=offset,
                limit=page_limit,
            ),
            empty_message="This album does not contain any media yet.",
        )

    def _new_album(self, *_args) -> None:
        name, ok = QInputDialog.getText(self, "New Album", "Album name")
        if not ok or not name.strip():
            return
        self.service.create_album(name, [])
        self.owner.refresh_views()

    def _delete_album(self, *_args) -> None:
        album_id = self._current_album_id()
        if not album_id:
            return
        self.service.delete_album(album_id)
        self.owner.refresh_views()

    def _current_album_id(self) -> str:
        current = self.album_list.currentItem()
        if not current:
            return ""
        return str(current.data(Qt.UserRole))

    def set_busy(self, busy: bool) -> None:
        self.new_album_button.setEnabled(not busy)
        self.delete_album_button.setEnabled(not busy)


class MemoriesPage(QWidget):
    def __init__(self, service: LibraryService, owner: "MainWindow") -> None:
        super().__init__()
        self.service = service
        self.owner = owner

        self.memory_list = QListWidget()
        self.memory_list.currentItemChanged.connect(self._show_memory_items)

        self.regenerate_button = QPushButton("Rebuild Memories")
        self.regenerate_button.clicked.connect(self._rebuild_memories)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.memory_list)
        left_layout.addWidget(self.regenerate_button)

        left_panel = QWidget()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(340)

        self.summary = QLabel("Select a memory")
        self.summary.setWordWrap(True)
        self.grid = MediaGridWidget(service)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.summary)
        right_layout.addWidget(self.grid)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)

        outer_layout = QHBoxLayout(self)
        outer_layout.addWidget(left_panel)
        outer_layout.addWidget(right_panel, 1)

    def refresh(self, *_args, load_items: bool = True) -> None:
        selected_id = self._current_memory_id()
        self.memory_list.blockSignals(True)
        self.memory_list.clear()
        for memory in self.service.list_memories():
            item = QListWidgetItem(f"{memory.title} ({len(memory.item_ids)})")
            item.setData(Qt.UserRole, memory.id)
            self.memory_list.addItem(item)
            if memory.id == selected_id:
                self.memory_list.setCurrentItem(item)
        self.memory_list.blockSignals(False)
        if self.memory_list.count() and not self.memory_list.currentItem():
            self.memory_list.setCurrentRow(0)
        if load_items:
            self._show_memory_items()

    def _show_memory_items(self, *_args) -> None:
        memory_id = self._current_memory_id()
        memory = next((entry for entry in self.service.list_memories() if entry.id == memory_id), None)
        if not memory:
            self.summary.setText("Select a memory")
            self.grid.set_items([])
            return
        self.summary.setText(f"{memory.title}\n{memory.subtitle}\n{memory.summary}")
        self.grid.set_page_loader(
            lambda offset, page_limit, memory_id=memory_id: self.service.items_for_memory_page(
                memory_id,
                offset=offset,
                limit=page_limit,
            ),
            empty_message="This memory does not contain any media yet.",
        )

    def _rebuild_memories(self, *_args) -> None:
        self.service.regenerate_memories()
        self.service.save()
        self.owner.refresh_views()

    def _current_memory_id(self) -> str:
        current = self.memory_list.currentItem()
        if not current:
            return ""
        return str(current.data(Qt.UserRole))

    def set_busy(self, busy: bool) -> None:
        self.regenerate_button.setEnabled(not busy)


class ModelsPage(QWidget):
    def __init__(self, service: LibraryService, owner: "MainWindow") -> None:
        super().__init__()
        self.service = service
        self.owner = owner

        self.model_list = QListWidget()
        self.model_list.currentItemChanged.connect(self._show_details)

        self.details = QTextEdit()
        self.details.setReadOnly(True)

        self.refresh_button = QPushButton("Refresh Model Status")
        self.refresh_button.clicked.connect(self.refresh)
        self.download_selected_button = QPushButton("Download Selected Model")
        self.download_selected_button.clicked.connect(self._download_selected_model)
        self.download_recommended_button = QPushButton("Download Recommended Models")
        self.download_recommended_button.clicked.connect(self._download_recommended_models)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.model_list)
        left_layout.addWidget(self.refresh_button)
        left_layout.addWidget(self.download_selected_button)
        left_layout.addWidget(self.download_recommended_button)

        left_panel = QWidget()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(360)

        outer_layout = QHBoxLayout(self)
        outer_layout.addWidget(left_panel)
        outer_layout.addWidget(self.details, 1)

    def refresh(self, *_args) -> None:
        selected_id = self._current_model_id()
        self.model_list.blockSignals(True)
        self.model_list.clear()
        for status in self.service.model_statuses():
            suffix = "installed" if status.installed else "missing"
            item = QListWidgetItem(f"{status.title} ({suffix})")
            item.setData(Qt.UserRole, status.id)
            self.model_list.addItem(item)
            if status.id == selected_id:
                self.model_list.setCurrentItem(item)
        self.model_list.blockSignals(False)
        if self.model_list.count() and not self.model_list.currentItem():
            self.model_list.setCurrentRow(0)
        self._show_details()

    def _show_details(self, *_args) -> None:
        model_id = self._current_model_id()
        if not model_id:
            self.details.setText("Select a model to view status and download details.")
            return
        status = next(
            (entry for entry in self.service.model_statuses() if entry.id == model_id),
            None,
        )
        if status is None:
            self.details.setText("Model status unavailable.")
            return
        install_text = "Installed" if status.installed else "Not downloaded"
        self.details.setText(
            "\n".join(
                [
                    status.title,
                    f"Role: {status.role}",
                    f"Status: {install_text}",
                    f"Local path: {status.local_path}",
                    "",
                    status.summary,
                    "",
                    f"Source: {status.source_url}",
                    f"Download: {status.download_url}",
                    "",
                    f"License notes: {status.license_note}",
                ]
            )
        )

    def _download_recommended_models(self, *_args) -> None:
        self.owner.download_recommended_models()

    def _download_selected_model(self, *_args) -> None:
        model_id = self._current_model_id()
        if not model_id:
            QMessageBox.warning(self, "No Model Selected", "Choose a model first.")
            return
        self.owner.download_model(model_id)

    def _current_model_id(self) -> str:
        current = self.model_list.currentItem()
        if not current:
            return ""
        return str(current.data(Qt.UserRole))

    def set_busy(self, busy: bool) -> None:
        self.refresh_button.setEnabled(not busy)
        self.download_selected_button.setEnabled(not busy)
        self.download_recommended_button.setEnabled(not busy)


class MainWindow(QMainWindow):
    def __init__(self, service: LibraryService) -> None:
        super().__init__()
        self.service = service
        self._task_thread: QThread | None = None
        self._task_worker: BackgroundTaskWorker | None = None
        self._cluster_cache_thread: QThread | None = None
        self._cluster_cache_worker: UnknownClusterCacheWorker | None = None
        self._cluster_cache_last_started = 0.0
        self._cluster_cache_final_pending = False
        self._main_task_base_message = ""
        self._main_task_progress: ProgressUpdate | None = None
        self._cluster_cache_base_message = ""
        self._cluster_cache_progress: ProgressUpdate | None = None
        self._active_task = ""
        self._live_refresh_pending = False
        self._library_dirty = False
        self._people_dirty = False
        self._unknown_clusters_dirty = True
        self._albums_dirty = False
        self._memories_dirty = False
        self._models_dirty = False
        self.setWindowTitle("Linux Smart Photos")
        self.resize(1560, 980)

        self.tabs = QTabWidget()
        self.library_page = LibraryPage(service, self)
        self.people_page = PeoplePage(service, self)
        self.unknown_clusters_page = UnknownClustersPage(service, self)
        self.albums_page = AlbumsPage(service, self)
        self.memories_page = MemoriesPage(service, self)
        self.models_page = ModelsPage(service, self)

        self.tabs.addTab(self.library_page, "Library")
        self.tabs.addTab(self.people_page, "People & Pets")
        self.tabs.addTab(self.unknown_clusters_page, "Unknown Clusters")
        self.tabs.addTab(self.albums_page, "Albums")
        self.tabs.addTab(self.memories_page, "Memories")
        self.tabs.addTab(self.models_page, "AI Models")
        self.tabs.currentChanged.connect(self._handle_tab_changed)
        self.setCentralWidget(self.tabs)

        self.status_panel = QWidget()
        self.status_panel_layout = QVBoxLayout(self.status_panel)
        self.status_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.status_panel_layout.setSpacing(2)
        self.main_task_status = TaskStatusRow("Background Task")
        self.cluster_cache_status = TaskStatusRow("Cluster Cache")
        self.cluster_view_status = TaskStatusRow("Cluster View")
        self.status_panel_layout.addWidget(self.main_task_status)
        self.status_panel_layout.addWidget(self.cluster_cache_status)
        self.status_panel_layout.addWidget(self.cluster_view_status)
        self.statusBar().addPermanentWidget(self.status_panel, 1)
        self.main_task_status.set_idle("Ready")
        self.cluster_cache_status.set_idle("Waiting for scan progress")
        self.cluster_view_status.set_idle("Unknown clusters tab idle")
        self._status_refresh_timer = QTimer(self)
        self._status_refresh_timer.setInterval(1000)
        self._status_refresh_timer.timeout.connect(self._refresh_active_status_rows)
        self._status_refresh_timer.start()

        QTimer.singleShot(0, self._finish_startup)

    def _finish_startup(self) -> None:
        self.refresh_views(refresh_unknown=False)
        QTimer.singleShot(0, self._start_startup_tasks)

    def refresh_views(
        self,
        refresh_unknown: bool | None = None,
        library_limit: int | None = None,
    ) -> None:
        current_widget = self.tabs.currentWidget()
        self.library_page._refresh_persona_filter()
        if current_widget is self.library_page:
            self.library_page.refresh()
            self._library_dirty = False
        else:
            self._library_dirty = True

        self.people_page.refresh(load_items=current_widget is self.people_page)
        self._people_dirty = current_widget is not self.people_page

        self.albums_page.refresh(load_items=current_widget is self.albums_page)
        self._albums_dirty = current_widget is not self.albums_page

        self.memories_page.refresh(load_items=current_widget is self.memories_page)
        self._memories_dirty = current_widget is not self.memories_page

        self.models_page.refresh()
        self._models_dirty = False
        should_refresh_unknown = (
            current_widget is self.unknown_clusters_page
            if refresh_unknown is None
            else refresh_unknown
        )
        if should_refresh_unknown:
            self.unknown_clusters_page.refresh()
            self._unknown_clusters_dirty = False
        else:
            self._unknown_clusters_dirty = True

    def refresh_live_views(self) -> None:
        current_widget = self.tabs.currentWidget()

        if current_widget is self.library_page:
            self.library_page._refresh_persona_filter()
            self.library_page.refresh()
            self._library_dirty = False
        else:
            self._library_dirty = True

        if current_widget is self.people_page:
            self.people_page.refresh()
            self._people_dirty = False
        else:
            self._people_dirty = True

        if current_widget is self.unknown_clusters_page:
            self.unknown_clusters_page.refresh()
            self._unknown_clusters_dirty = False
        else:
            self._unknown_clusters_dirty = True

        self._albums_dirty = True
        self._memories_dirty = True
        self._models_dirty = True

    def _start_startup_tasks(self) -> None:
        self._start_background_task("startup")

    def sync_library(self, *_args) -> None:
        self._start_background_task("sync")

    def download_recommended_models(self) -> None:
        self._start_background_task("download_recommended_models")

    def download_model(self, model_id: str) -> None:
        self._start_background_task("download_model", model_id=model_id)

    def _start_background_task(self, task_name: str, model_id: str = "") -> bool:
        if self._task_thread is not None:
            self.main_task_status.set_error("A background task is already running.")
            return False

        self._active_task = task_name
        self._set_busy(True, self._task_start_message(task_name, model_id))

        thread = QThread(self)
        worker = BackgroundTaskWorker(self.service.config, task_name, model_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._handle_task_progress)
        worker.completed.connect(self._handle_task_completed)
        worker.failed.connect(self._handle_task_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_task_handles)

        self._task_thread = thread
        self._task_worker = worker
        thread.start()
        return True

    def _handle_task_progress(self, update: Any) -> None:
        if not isinstance(update, ProgressUpdate):
            return

        detail = f": {update.detail}" if update.detail else ""
        self._main_task_base_message = f"{update.message}{detail}"
        self._main_task_progress = update
        self._apply_progress_to_row(
            self.main_task_status,
            self._main_task_base_message,
            update,
        )
        if update.snapshot_ready:
            self._schedule_live_refresh()
            if self._active_task in {"startup", "sync"}:
                self._maybe_refresh_unknown_cluster_cache()

    def _handle_task_completed(self, payload: Any) -> None:
        task_name = str(payload.get("task", self._active_task))
        self.service.reload()
        self.refresh_views()
        self._set_busy(False, self._task_success_message(task_name, payload))
        self._main_task_progress = None
        self._main_task_base_message = ""
        if task_name in {"startup", "sync"}:
            self._start_unknown_cluster_cache_refresh(partial=False)
        if task_name == "startup" and payload.get("download_error"):
            QMessageBox.warning(
                self,
                "AI Model Download Incomplete",
                f"{payload['download_error']}\n\nThe library scan still completed. "
                "Open the AI Models tab to retry the missing downloads.",
            )

    def _handle_task_failed(self, message: str) -> None:
        task_name = self._active_task
        self.service.reload()
        self.refresh_views()
        self._set_busy(False, self._task_failure_message(task_name))
        self._main_task_progress = None
        self._main_task_base_message = ""
        self.main_task_status.set_error(f"{self._task_failure_message(task_name)}: {message}")
        QMessageBox.critical(self, "Background Task Failed", message)

    def _handle_tab_changed(self, _index: int) -> None:
        current_widget = self.tabs.currentWidget()
        if current_widget is self.library_page and self._library_dirty:
            self.library_page._refresh_persona_filter()
            self.library_page.refresh()
            self._library_dirty = False
        if current_widget is self.people_page and self._people_dirty:
            self.people_page.refresh(load_items=True)
            self._people_dirty = False
        if current_widget is self.unknown_clusters_page and self._unknown_clusters_dirty:
            self.unknown_clusters_page.refresh()
            self._unknown_clusters_dirty = False
        if current_widget is self.albums_page and self._albums_dirty:
            self.albums_page.refresh(load_items=True)
            self._albums_dirty = False
        if current_widget is self.memories_page and self._memories_dirty:
            self.memories_page.refresh(load_items=True)
            self._memories_dirty = False
        if current_widget is self.models_page and self._models_dirty:
            self.models_page.refresh()
            self._models_dirty = False

    def _clear_task_handles(self) -> None:
        self._task_thread = None
        self._task_worker = None
        self._active_task = ""
        self._live_refresh_pending = False

    def _schedule_live_refresh(self) -> None:
        if self._live_refresh_pending:
            return
        self._live_refresh_pending = True
        QTimer.singleShot(LIVE_REFRESH_INTERVAL_MS, self._apply_live_refresh)

    def _apply_live_refresh(self) -> None:
        self._live_refresh_pending = False
        if not self._active_task:
            return
        self.service.reload()
        self.refresh_live_views()

    def _maybe_refresh_unknown_cluster_cache(self) -> None:
        if self._cluster_cache_thread is not None:
            return
        if monotonic() - self._cluster_cache_last_started < 5.0:
            return
        self._start_unknown_cluster_cache_refresh(partial=True)

    def _start_unknown_cluster_cache_refresh(self, *, partial: bool) -> None:
        if self._cluster_cache_thread is not None:
            if not partial:
                self._cluster_cache_final_pending = True
            return

        thread = QThread(self)
        worker = UnknownClusterCacheWorker(self.service.config, partial=partial)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._handle_unknown_cluster_cache_progress)
        worker.completed.connect(self._handle_unknown_cluster_cache_completed)
        worker.failed.connect(self._handle_unknown_cluster_cache_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_unknown_cluster_cache_handles)

        self._cluster_cache_thread = thread
        self._cluster_cache_worker = worker
        self._cluster_cache_last_started = monotonic()
        message = (
            "Building partial unknown-cluster cache from analyzed items"
            if partial
            else "Building final unknown-cluster cache"
        )
        self.cluster_cache_status.set_active(message, indeterminate=True)
        thread.start()

    def _handle_unknown_cluster_cache_completed(self, partial: bool) -> None:
        self._cluster_cache_progress = None
        self._cluster_cache_base_message = ""
        if partial:
            self.cluster_cache_status.set_complete("Partial unknown-cluster cache updated")
        else:
            self.cluster_cache_status.set_complete("Final unknown-cluster cache updated")
        current_widget = self.tabs.currentWidget()
        if current_widget is self.unknown_clusters_page:
            self.unknown_clusters_page.refresh()
            self._unknown_clusters_dirty = False
        else:
            self._unknown_clusters_dirty = True

    def _handle_unknown_cluster_cache_failed(self, message: str) -> None:
        self._cluster_cache_progress = None
        self._cluster_cache_base_message = ""
        self.cluster_cache_status.set_error(f"Cluster cache refresh failed: {message}")
        self._unknown_clusters_dirty = True

    def _handle_unknown_cluster_cache_progress(self, update: Any) -> None:
        if not isinstance(update, ProgressUpdate):
            return
        detail = f": {update.detail}" if update.detail else ""
        self._cluster_cache_base_message = f"{update.message}{detail}"
        self._cluster_cache_progress = update
        self._apply_progress_to_row(
            self.cluster_cache_status,
            self._cluster_cache_base_message,
            update,
        )

    def _clear_unknown_cluster_cache_handles(self) -> None:
        self._cluster_cache_thread = None
        self._cluster_cache_worker = None
        if self._cluster_cache_final_pending:
            self._cluster_cache_final_pending = False
            self._start_unknown_cluster_cache_refresh(partial=False)
        elif not self._active_task:
            self._cluster_cache_progress = None
            self._cluster_cache_base_message = ""
            self.cluster_cache_status.set_idle("Waiting for scan progress")

    def set_cluster_view_status_loading(self, message: str) -> None:
        self.cluster_view_status.set_active(message, indeterminate=True)

    def set_cluster_view_status_complete(self, message: str) -> None:
        self.cluster_view_status.set_complete(message)

    def set_cluster_view_status_error(self, message: str) -> None:
        self.cluster_view_status.set_error(message)

    def _format_progress_status(self, base_message: str, update: ProgressUpdate) -> str:
        fragments: list[str] = []
        step_seconds, elapsed_seconds, eta_seconds = self._effective_progress_timing(update)
        if step_seconds is not None:
            fragments.append(f"batch {self._format_duration(step_seconds)}")
        if elapsed_seconds is not None:
            fragments.append(f"elapsed {self._format_duration(elapsed_seconds)}")
        if eta_seconds is not None:
            fragments.append(f"ETA {self._format_duration(eta_seconds)}")
        if not fragments:
            return base_message
        return f"{base_message}  |  " + "  |  ".join(fragments)

    def _format_duration(self, seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        if minutes > 0:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    def _apply_progress_to_row(
        self,
        row: TaskStatusRow,
        base_message: str,
        update: ProgressUpdate,
    ) -> None:
        row.set_active(
            self._format_progress_status(base_message, update),
            current=update.current,
            total=update.total,
            indeterminate=update.indeterminate,
        )

    def _effective_progress_timing(
        self,
        update: ProgressUpdate,
    ) -> tuple[float | None, float | None, float | None]:
        step_seconds = update.step_seconds
        elapsed_seconds = update.elapsed_seconds
        eta_seconds = update.eta_seconds
        if update.timestamp_seconds is None:
            return step_seconds, elapsed_seconds, eta_seconds
        delta = max(0.0, monotonic() - update.timestamp_seconds)
        if step_seconds is not None:
            step_seconds += delta
        if elapsed_seconds is not None:
            elapsed_seconds += delta
        if eta_seconds is not None:
            eta_seconds = max(0.0, eta_seconds + delta)
        return step_seconds, elapsed_seconds, eta_seconds

    def _refresh_active_status_rows(self) -> None:
        if self._main_task_progress is not None and self._main_task_base_message:
            self._apply_progress_to_row(
                self.main_task_status,
                self._main_task_base_message,
                self._main_task_progress,
            )
        if self._cluster_cache_progress is not None and self._cluster_cache_base_message:
            self._apply_progress_to_row(
                self.cluster_cache_status,
                self._cluster_cache_base_message,
                self._cluster_cache_progress,
            )

    def _set_busy(self, busy: bool, message: str) -> None:
        for page in (
            self.library_page,
            self.people_page,
            self.unknown_clusters_page,
            self.albums_page,
            self.memories_page,
            self.models_page,
        ):
            page.set_busy(busy)

        if busy:
            self.main_task_status.set_active(message, indeterminate=True)
            if self._active_task in {"startup", "sync"} and self._cluster_cache_thread is None:
                self.cluster_cache_status.set_idle("Waiting for analyzed batches")
        else:
            self.main_task_status.set_complete(message)

    def _task_start_message(self, task_name: str, model_id: str) -> str:
        if task_name == "startup":
            return "Preparing AI models and scanning the library."
        if task_name == "sync":
            return "Scanning the library in the background."
        if task_name == "download_recommended_models":
            return "Downloading recommended AI models."
        if task_name == "download_model":
            return f"Downloading AI model: {model_id}"
        return "Running background task."

    def _task_success_message(self, task_name: str, payload: dict[str, Any]) -> str:
        if task_name in {"startup", "sync"}:
            summary = payload.get("summary")
            if summary is not None:
                return (
                    f"Sync complete: {summary.added} added, "
                    f"{summary.updated} updated, {summary.removed} removed"
                )
        if task_name == "download_recommended_models":
            missing_model_ids = payload.get("missing_model_ids", [])
            if missing_model_ids:
                return f"Downloaded {len(missing_model_ids)} recommended models."
            return "Recommended AI models are already installed."
        if task_name == "download_model":
            path = str(payload.get("path", ""))
            return f"Model ready: {path}" if path else "Selected model is ready."
        return "Background task complete."

    def _task_failure_message(self, task_name: str) -> str:
        if task_name == "startup":
            return "Startup task failed."
        if task_name == "sync":
            return "Sync failed."
        if task_name == "download_recommended_models":
            return "Recommended model download failed."
        if task_name == "download_model":
            return "Model download failed."
        return "Background task failed."

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._task_thread is not None or self._cluster_cache_thread is not None:
            QMessageBox.information(
                self,
                "Background Task Running",
                "Wait for the current background task to finish before closing Smart Photos.",
            )
            event.ignore()
            return
        super().closeEvent(event)
