from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..services.library import LibraryService
from .dialogs import AlbumDialog, CorrectionsDialog
from .widgets import MediaGridWidget


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

        refresh_button = QPushButton("Refresh Library")
        refresh_button.clicked.connect(self.owner.sync_library)
        correct_button = QPushButton("Assign / Correct")
        correct_button.clicked.connect(self.open_corrections)
        album_button = QPushButton("Add To Album")
        album_button.clicked.connect(self.add_to_album)
        favorite_button = QPushButton("Toggle Favorite")
        favorite_button.clicked.connect(self.toggle_favorite)

        controls.addWidget(self.search_box)
        controls.addWidget(self.type_filter)
        controls.addWidget(self.kind_filter)
        controls.addWidget(self.persona_filter)
        controls.addWidget(self.favorites_only)
        controls.addStretch(1)
        controls.addWidget(correct_button)
        controls.addWidget(album_button)
        controls.addWidget(favorite_button)
        controls.addWidget(refresh_button)

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

    def refresh(self, *_args) -> None:
        items = self.service.search_items(
            query=self.search_box.text(),
            media_kind=str(self.type_filter.currentData()),
            persona_kind=str(self.kind_filter.currentData()),
            persona_id=str(self.persona_filter.currentData()),
            favorites_only=self.favorites_only.isChecked(),
        )
        self.grid.set_items(items)

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


class PeoplePage(QWidget):
    def __init__(self, service: LibraryService, owner: "MainWindow") -> None:
        super().__init__()
        self.service = service
        self.owner = owner

        self.kind_filter = QComboBox()
        self.kind_filter.addItem("All Personas", "all")
        self.kind_filter.addItem("People", "person")
        self.kind_filter.addItem("Pets", "pet")
        self.kind_filter.currentIndexChanged.connect(self.refresh)

        self.persona_list = QListWidget()
        self.persona_list.currentItemChanged.connect(self._show_persona_items)

        new_persona_button = QPushButton("New Persona")
        new_persona_button.clicked.connect(self._create_persona)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.kind_filter)
        left_layout.addWidget(self.persona_list)
        left_layout.addWidget(new_persona_button)

        left_panel = QWidget()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(320)

        self.grid = MediaGridWidget(service)
        correct_button = QPushButton("Correct Current Selection")
        correct_button.clicked.connect(self._open_corrections)

        right_layout = QVBoxLayout()
        right_layout.addWidget(correct_button)
        right_layout.addWidget(self.grid)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)

        outer_layout = QHBoxLayout(self)
        outer_layout.addWidget(left_panel)
        outer_layout.addWidget(right_panel, 1)

    def refresh(self, *_args) -> None:
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
        self._show_persona_items()

    def _show_persona_items(self, *_args) -> None:
        persona_id = self._current_persona_id()
        self.grid.set_items(self.service.items_for_persona(persona_id) if persona_id else [])

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


class AlbumsPage(QWidget):
    def __init__(self, service: LibraryService, owner: "MainWindow") -> None:
        super().__init__()
        self.service = service
        self.owner = owner

        self.album_list = QListWidget()
        self.album_list.currentItemChanged.connect(self._show_album_items)

        new_album_button = QPushButton("New Empty Album")
        new_album_button.clicked.connect(self._new_album)
        delete_album_button = QPushButton("Delete Album")
        delete_album_button.clicked.connect(self._delete_album)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.album_list)
        left_layout.addWidget(new_album_button)
        left_layout.addWidget(delete_album_button)

        left_panel = QWidget()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(320)

        self.grid = MediaGridWidget(service)

        outer_layout = QHBoxLayout(self)
        outer_layout.addWidget(left_panel)
        outer_layout.addWidget(self.grid, 1)

    def refresh(self, *_args) -> None:
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
        self._show_album_items()

    def _show_album_items(self, *_args) -> None:
        album_id = self._current_album_id()
        self.grid.set_items(self.service.items_for_album(album_id) if album_id else [])

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


class MemoriesPage(QWidget):
    def __init__(self, service: LibraryService, owner: "MainWindow") -> None:
        super().__init__()
        self.service = service
        self.owner = owner

        self.memory_list = QListWidget()
        self.memory_list.currentItemChanged.connect(self._show_memory_items)

        regenerate_button = QPushButton("Rebuild Memories")
        regenerate_button.clicked.connect(self._rebuild_memories)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.memory_list)
        left_layout.addWidget(regenerate_button)

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

    def refresh(self, *_args) -> None:
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
        self._show_memory_items()

    def _show_memory_items(self, *_args) -> None:
        memory_id = self._current_memory_id()
        memory = self.service.state.memories.get(memory_id)
        if not memory:
            self.summary.setText("Select a memory")
            self.grid.set_items([])
            return
        self.summary.setText(f"{memory.title}\n{memory.subtitle}\n{memory.summary}")
        self.grid.set_items(self.service.items_for_memory(memory_id))

    def _rebuild_memories(self, *_args) -> None:
        self.service.regenerate_memories()
        self.service.save()
        self.owner.refresh_views()

    def _current_memory_id(self) -> str:
        current = self.memory_list.currentItem()
        if not current:
            return ""
        return str(current.data(Qt.UserRole))


class ModelsPage(QWidget):
    def __init__(self, service: LibraryService, owner: "MainWindow") -> None:
        super().__init__()
        self.service = service
        self.owner = owner

        self.model_list = QListWidget()
        self.model_list.currentItemChanged.connect(self._show_details)

        self.details = QTextEdit()
        self.details.setReadOnly(True)

        refresh_button = QPushButton("Refresh Model Status")
        refresh_button.clicked.connect(self.refresh)
        download_selected_button = QPushButton("Download Selected Model")
        download_selected_button.clicked.connect(self._download_selected_model)
        download_recommended_button = QPushButton("Download Recommended Models")
        download_recommended_button.clicked.connect(self._download_recommended_models)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.model_list)
        left_layout.addWidget(refresh_button)
        left_layout.addWidget(download_selected_button)
        left_layout.addWidget(download_recommended_button)

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
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            paths = self.service.download_recommended_models()
            self.owner.statusBar().showMessage(f"Downloaded {len(paths)} recommended models.")
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "Model Download Failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _download_selected_model(self, *_args) -> None:
        model_id = self._current_model_id()
        if not model_id:
            QMessageBox.warning(self, "No Model Selected", "Choose a model first.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            path = self.service.download_model(model_id)
            self.owner.statusBar().showMessage(f"Downloaded model to {path}")
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "Model Download Failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _current_model_id(self) -> str:
        current = self.model_list.currentItem()
        if not current:
            return ""
        return str(current.data(Qt.UserRole))


class MainWindow(QMainWindow):
    def __init__(self, service: LibraryService) -> None:
        super().__init__()
        self.service = service
        self.setWindowTitle("Linux Smart Photos")
        self.resize(1560, 980)

        self.tabs = QTabWidget()
        self.library_page = LibraryPage(service, self)
        self.people_page = PeoplePage(service, self)
        self.albums_page = AlbumsPage(service, self)
        self.memories_page = MemoriesPage(service, self)
        self.models_page = ModelsPage(service, self)

        self.tabs.addTab(self.library_page, "Library")
        self.tabs.addTab(self.people_page, "People & Pets")
        self.tabs.addTab(self.albums_page, "Albums")
        self.tabs.addTab(self.memories_page, "Memories")
        self.tabs.addTab(self.models_page, "AI Models")
        self.setCentralWidget(self.tabs)

        self.statusBar().showMessage("Ready")
        self.refresh_views()
        QTimer.singleShot(0, self.sync_library)

    def refresh_views(self) -> None:
        self.library_page._refresh_persona_filter()
        self.library_page.refresh()
        self.people_page.refresh()
        self.albums_page.refresh()
        self.memories_page.refresh()
        self.models_page.refresh()

    def sync_library(self, *_args) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            summary = self.service.sync()
            self.statusBar().showMessage(
                f"Sync complete: {summary.added} added, {summary.updated} updated, {summary.removed} removed"
            )
            self.refresh_views()
        except Exception as exc:
            QMessageBox.critical(self, "Sync Failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
