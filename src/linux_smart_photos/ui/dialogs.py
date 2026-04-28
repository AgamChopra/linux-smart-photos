from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class AssignPersonaDialog(QDialog):
    def __init__(
        self,
        personas: list,
        suggested_kind: str = "person",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Assign Persona")
        self.personas = personas

        self.kind_combo = QComboBox()
        self.kind_combo.addItem("Person", "person")
        self.kind_combo.addItem("Pet", "pet")
        self.kind_combo.setCurrentIndex(0 if suggested_kind != "pet" else 1)
        self.kind_combo.currentIndexChanged.connect(self._populate_personas)

        self.persona_list = QListWidget()
        self.new_name_edit = QLineEdit()
        self.new_name_edit.setPlaceholderText("Or create a new persona")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Kind"))
        layout.addWidget(self.kind_combo)
        layout.addWidget(QLabel("Existing Personas"))
        layout.addWidget(self.persona_list)
        layout.addWidget(QLabel("New Persona"))
        layout.addWidget(self.new_name_edit)
        layout.addWidget(buttons)

        self._populate_personas()

    def selection(self) -> dict[str, str]:
        current = self.persona_list.currentItem()
        return {
            "persona_id": str(current.data(Qt.UserRole)) if current else "",
            "new_name": self.new_name_edit.text().strip(),
            "kind": str(self.kind_combo.currentData()),
        }

    def accept(self) -> None:
        choice = self.selection()
        if not choice["persona_id"] and not choice["new_name"]:
            QMessageBox.warning(self, "Persona Required", "Choose an existing persona or enter a new name.")
            return
        super().accept()

    def _populate_personas(self, *_args) -> None:
        current_kind = str(self.kind_combo.currentData())
        self.persona_list.clear()
        for persona in self.personas:
            if persona.kind != current_kind:
                continue
            item = QListWidgetItem(persona.name)
            item.setData(Qt.UserRole, persona.id)
            self.persona_list.addItem(item)


class AlbumDialog(QDialog):
    def __init__(self, albums: list, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add To Album")
        self.albums = albums

        self.album_list = QListWidget()
        for album in albums:
            item = QListWidgetItem(album.name)
            item.setData(Qt.UserRole, album.id)
            self.album_list.addItem(item)

        self.new_name_edit = QLineEdit()
        self.new_name_edit.setPlaceholderText("Or create a new album")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Existing Albums"))
        layout.addWidget(self.album_list)
        layout.addWidget(QLabel("New Album"))
        layout.addWidget(self.new_name_edit)
        layout.addWidget(buttons)

    def selection(self) -> dict[str, str]:
        current = self.album_list.currentItem()
        return {
            "album_id": str(current.data(Qt.UserRole)) if current else "",
            "new_name": self.new_name_edit.text().strip(),
        }

    def accept(self) -> None:
        choice = self.selection()
        if not choice["album_id"] and not choice["new_name"]:
            QMessageBox.warning(self, "Album Required", "Choose an album or enter a new album name.")
            return
        super().accept()


class CorrectionsDialog(QDialog):
    def __init__(self, service, item_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.item_id = item_id
        self.setWindowTitle("Correct People And Pets")
        self.resize(860, 620)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(240)
        self.preview_label.setObjectName("PreviewPanel")

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Region", "Label", "Assigned Persona", "Confidence"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        assign_region_button = QPushButton("Assign Selected Region")
        assign_region_button.clicked.connect(self._assign_region)
        clear_region_button = QPushButton("Clear Selected Region")
        clear_region_button.clicked.connect(self._clear_region)
        assign_whole_button = QPushButton("Assign Whole Asset")
        assign_whole_button.clicked.connect(self._assign_whole_asset)
        clear_whole_button = QPushButton("Clear Whole-Asset Personas")
        clear_whole_button.clicked.connect(self._clear_whole_asset)

        button_row = QHBoxLayout()
        button_row.addWidget(assign_region_button)
        button_row.addWidget(clear_region_button)
        button_row.addStretch(1)
        button_row.addWidget(assign_whole_button)
        button_row.addWidget(clear_whole_button)

        close_buttons = QDialogButtonBox(QDialogButtonBox.Close)
        close_buttons.rejected.connect(self.reject)
        close_buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self.preview_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.table)
        layout.addLayout(button_row)
        layout.addWidget(close_buttons)

        self.refresh()

    def refresh(self) -> None:
        item = self.service.state.items.get(self.item_id)
        if not item:
            self.reject()
            return

        pixmap = QPixmap(item.thumbnail_path) if item.thumbnail_path and Path(item.thumbnail_path).exists() else QPixmap()
        if not pixmap.isNull():
            self.preview_label.setPixmap(
                pixmap.scaled(640, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(item.title)

        persona_names = ", ".join(
            self.service.state.personas[persona_id].name
            for persona_id in item.manual_persona_ids
            if persona_id in self.service.state.personas
        ) or "None"
        self.summary_label.setText(
            f"{item.title}\nWhole-asset personas: {persona_names}\n"
            f"Detected regions: {len(item.detections)}"
        )

        self.table.setRowCount(0)
        for row_index, detection in enumerate(item.detections):
            self.table.insertRow(row_index)
            assigned_name = ""
            if detection.persona_id and detection.persona_id in self.service.state.personas:
                assigned_name = self.service.state.personas[detection.persona_id].name
            self.table.setItem(row_index, 0, QTableWidgetItem(detection.id))
            self.table.setItem(row_index, 1, QTableWidgetItem(detection.label))
            self.table.setItem(row_index, 2, QTableWidgetItem(assigned_name))
            self.table.setItem(row_index, 3, QTableWidgetItem(f"{detection.confidence:.2f}"))

    def _current_region_id(self) -> str:
        row = self.table.currentRow()
        if row < 0:
            return ""
        region_item = self.table.item(row, 0)
        return region_item.text() if region_item else ""

    def _assign_region(self) -> None:
        item = self.service.state.items.get(self.item_id)
        if not item:
            return
        region_id = self._current_region_id()
        if not region_id:
            QMessageBox.warning(self, "No Region Selected", "Select a detected face or object first.")
            return
        detection = next((entry for entry in item.detections if entry.id == region_id), None)
        if not detection:
            return
        suggested_kind = (
            "pet"
            if detection.kind.startswith("pet")
            or detection.label.lower() in {"cat", "kitten", "dog", "puppy", "bird", "horse", "rabbit", "pet"}
            else "person"
        )
        dialog = AssignPersonaDialog(self.service.list_personas(), suggested_kind=suggested_kind, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        choice = dialog.selection()
        try:
            self.service.assign_region_to_persona(
                self.item_id,
                region_id,
                persona_id=choice["persona_id"],
                new_name=choice["new_name"],
                kind=choice["kind"],
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Assignment Failed", str(exc))
            return
        self.refresh()

    def _clear_region(self) -> None:
        region_id = self._current_region_id()
        if not region_id:
            QMessageBox.warning(self, "No Region Selected", "Select a detected face or object first.")
            return
        self.service.clear_region_assignment(self.item_id, region_id)
        self.refresh()

    def _assign_whole_asset(self) -> None:
        dialog = AssignPersonaDialog(self.service.list_personas(), parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        choice = dialog.selection()
        self.service.assign_item_to_persona(
            self.item_id,
            persona_id=choice["persona_id"],
            new_name=choice["new_name"],
            kind=choice["kind"],
        )
        self.refresh()

    def _clear_whole_asset(self) -> None:
        self.service.clear_item_personas(self.item_id)
        self.refresh()
