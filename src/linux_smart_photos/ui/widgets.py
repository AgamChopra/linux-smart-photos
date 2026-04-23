from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QIcon, QMovie, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..media import VIDEO_EXTENSIONS
from ..models import MediaItem
from ..services.library import LibraryService


QAudioOutput = None
QMediaPlayer = None
QVideoWidget = None
_MULTIMEDIA_IMPORT_ATTEMPTED = False


def _load_multimedia_backend():
    global QAudioOutput, QMediaPlayer, QVideoWidget, _MULTIMEDIA_IMPORT_ATTEMPTED

    if _MULTIMEDIA_IMPORT_ATTEMPTED:
        return QAudioOutput, QMediaPlayer, QVideoWidget

    _MULTIMEDIA_IMPORT_ATTEMPTED = True
    try:
        from PySide6.QtMultimedia import QAudioOutput as _QAudioOutput, QMediaPlayer as _QMediaPlayer
        from PySide6.QtMultimediaWidgets import QVideoWidget as _QVideoWidget
    except Exception:
        return None, None, None

    QAudioOutput = _QAudioOutput
    QMediaPlayer = _QMediaPlayer
    QVideoWidget = _QVideoWidget
    return QAudioOutput, QMediaPlayer, QVideoWidget


class MediaGridWidget(QWidget):
    POPULATE_CHUNK_SIZE = 96

    selectionChanged = Signal(list)

    def __init__(self, service: LibraryService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self._items: dict[str, MediaItem] = {}
        self._movie: QMovie | None = None
        self._preview_video_path = ""

        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setMovement(QListWidget.Static)
        self.list_widget.setSpacing(12)
        self.list_widget.setIconSize(QSize(180, 180))
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.currentItemChanged.connect(self._update_preview)
        self.list_widget.itemSelectionChanged.connect(self._emit_selection)
        self._placeholder_icon = self.style().standardIcon(QStyle.SP_FileIcon)
        self._pending_items: list[MediaItem] = []
        self._populate_cursor = 0
        self._pending_previous_item_id = ""
        self._populate_timer = QTimer(self)
        self._populate_timer.setSingleShot(True)
        self._populate_timer.timeout.connect(self._populate_next_chunk)

        self.preview_label = QLabel("Select a media item")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumWidth(360)
        self.preview_label.setMinimumHeight(360)
        self.preview_label.setStyleSheet(
            "background: #f3f1ec; border: 1px solid #ddd3c7; border-radius: 12px;"
        )

        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(self.preview_label)

        self.video_status_label = QLabel("Video playback is unavailable in this environment.")
        self.video_status_label.setAlignment(Qt.AlignCenter)
        self.video_status_label.setMinimumWidth(360)
        self.video_status_label.setMinimumHeight(360)
        self.video_status_label.setStyleSheet(
            "background: #f3f1ec; border: 1px solid #ddd3c7; border-radius: 12px;"
        )

        self.video_widget = None
        self.media_player = None
        self.audio_output = None
        self.preview_stack.addWidget(self.video_status_label)

        self.video_toggle_button = QPushButton("Play Preview")
        self.video_toggle_button.clicked.connect(self._toggle_video)
        self.video_toggle_button.setVisible(False)

        preview_controls = QHBoxLayout()
        preview_controls.addStretch(1)
        preview_controls.addWidget(self.video_toggle_button)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumWidth(360)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.preview_stack)
        right_layout.addLayout(preview_controls)
        right_layout.addWidget(self.details)

        splitter = QSplitter()
        splitter.addWidget(self.list_widget)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def set_items(self, items: list[MediaItem]) -> None:
        previous_item_id = self.current_item_id()
        self._items = {item.id: item for item in items}
        self._clear_preview_media()
        self._populate_timer.stop()
        self.list_widget.clear()
        self._pending_items = list(items)
        self._populate_cursor = 0
        self._pending_previous_item_id = previous_item_id

        if not items:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("No media items match the current view")
            self.preview_stack.setCurrentWidget(self.preview_label)
            self.video_toggle_button.setVisible(False)
            self.details.setText("")
            return

        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText(f"Loading {len(items)} media items...")
        self.preview_stack.setCurrentWidget(self.preview_label)
        self.video_toggle_button.setVisible(False)
        self.details.setText("")
        self._populate_timer.start(0)

    def current_item_id(self) -> str:
        current = self.list_widget.currentItem()
        if not current:
            return ""
        return str(current.data(Qt.UserRole))

    def selected_item_ids(self) -> list[str]:
        return [
            str(item.data(Qt.UserRole))
            for item in self.list_widget.selectedItems()
        ]

    def _emit_selection(self) -> None:
        self.selectionChanged.emit(self.selected_item_ids())

    def _update_preview(self, *_args) -> None:
        self._clear_preview_media()
        item_id = self.current_item_id()
        item = self._items.get(item_id)
        if not item:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Select a media item")
            self.preview_stack.setCurrentWidget(self.preview_label)
            self.video_toggle_button.setVisible(False)
            self.details.setText("")
            return

        if item.media_kind in {"video", "live_photo"}:
            self._show_video_preview(item)
        elif item.media_kind == "gif":
            self._show_gif_preview(item)
        else:
            self._show_image_preview(item)
        self.details.setText(self.service.build_item_details(item))

    def _show_image_preview(self, item: MediaItem) -> None:
        pixmap = QPixmap(item.thumbnail_path) if item.thumbnail_path else QPixmap()
        if not pixmap.isNull():
            self.preview_label.setPixmap(
                pixmap.scaled(
                    420,
                    420,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            self.preview_label.setText("")
        else:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(item.title)
        self.preview_stack.setCurrentWidget(self.preview_label)
        self.video_toggle_button.setVisible(False)

    def _show_gif_preview(self, item: MediaItem) -> None:
        if Path(item.path).exists():
            movie = QMovie(item.path)
            if movie.isValid():
                movie.setScaledSize(QSize(420, 420))
                self.preview_label.setMovie(movie)
                movie.start()
                self._movie = movie
                self.preview_label.setText("")
                self.preview_stack.setCurrentWidget(self.preview_label)
                self.video_toggle_button.setVisible(False)
                return
        self._show_image_preview(item)

    def _show_video_preview(self, item: MediaItem) -> None:
        video_path = self._video_path_for_item(item)
        if not video_path:
            self._show_image_preview(item)
            return

        self._preview_video_path = video_path
        self._show_image_preview(item)
        self.video_toggle_button.setText("Play Preview")
        self.video_toggle_button.setVisible(True)

    def _clear_preview_media(self) -> None:
        self._preview_video_path = ""
        if self._movie is not None:
            self._movie.stop()
            self._movie = None
        self.preview_label.clear()
        if self.media_player is not None:
            self.media_player.stop()
            self.media_player.setSource(QUrl())

    def _toggle_video(self) -> None:
        if not self._preview_video_path:
            return

        if not self._ensure_video_backend():
            self.video_status_label.setText(
                f"Video preview needs Qt multimedia support.\n\n{self._preview_video_path}"
            )
            self.preview_stack.setCurrentWidget(self.video_status_label)
            self.video_toggle_button.setText("Play Preview")
            return

        target_url = QUrl.fromLocalFile(self._preview_video_path)
        if self.preview_stack.currentWidget() is not self.video_widget or self.media_player.source() != target_url:
            self.preview_stack.setCurrentWidget(self.video_widget)
            self.media_player.setSource(target_url)
            self.media_player.play()
            self.video_toggle_button.setText("Pause Preview")
            return

        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.video_toggle_button.setText("Play Preview")
        else:
            self.media_player.play()
            self.video_toggle_button.setText("Pause Preview")

    def _video_path_for_item(self, item: MediaItem) -> str:
        for component in item.component_paths:
            if Path(component).suffix.lower() in VIDEO_EXTENSIONS and Path(component).exists():
                return component
        if Path(item.path).suffix.lower() in VIDEO_EXTENSIONS and Path(item.path).exists():
            return item.path
        return ""

    def _ensure_video_backend(self) -> bool:
        if self.media_player is not None and self.video_widget is not None:
            return True

        audio_output_cls, media_player_cls, video_widget_cls = _load_multimedia_backend()
        if audio_output_cls is None or media_player_cls is None or video_widget_cls is None:
            return False

        try:
            self.video_widget = video_widget_cls()
            self.media_player = media_player_cls(self)
            self.audio_output = audio_output_cls(self)
            self.audio_output.setVolume(0.0)
            self.media_player.setAudioOutput(self.audio_output)
            self.media_player.setVideoOutput(self.video_widget)
            self.preview_stack.addWidget(self.video_widget)
            return True
        except Exception:
            self.video_widget = None
            self.media_player = None
            self.audio_output = None
            return False

    def _populate_next_chunk(self) -> None:
        end_index = min(
            len(self._pending_items),
            self._populate_cursor + self.POPULATE_CHUNK_SIZE,
        )
        for item in self._pending_items[self._populate_cursor:end_index]:
            icon = self._placeholder_icon
            thumbnail_path = Path(item.thumbnail_path)
            if item.thumbnail_path and thumbnail_path.exists():
                icon = QIcon(str(thumbnail_path))
            list_item = QListWidgetItem(icon, item.title)
            list_item.setData(Qt.UserRole, item.id)
            list_item.setToolTip(item.path)
            self.list_widget.addItem(list_item)

        self._populate_cursor = end_index
        if self._populate_cursor < len(self._pending_items):
            self.preview_label.setText(
                f"Loading {self._populate_cursor} / {len(self._pending_items)} media items..."
            )
            self._populate_timer.start(0)
            return

        if self._pending_previous_item_id:
            for index in range(self.list_widget.count()):
                list_item = self.list_widget.item(index)
                if str(list_item.data(Qt.UserRole)) == self._pending_previous_item_id:
                    self.list_widget.setCurrentItem(list_item)
                    break

        if not self.list_widget.currentItem():
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Select a media item")
            self.preview_stack.setCurrentWidget(self.preview_label)
            self.video_toggle_button.setVisible(False)
            self.details.setText("")
