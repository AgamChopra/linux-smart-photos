from __future__ import annotations

from bisect import bisect_left
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from time import monotonic
from typing import Callable, Iterator

try:
    import cv2
except Exception:
    cv2 = None

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
except Exception:
    register_heif_opener = None
else:
    register_heif_opener()

from PySide6.QtCore import QObject, QEvent, QPoint, QRunnable, QRect, QSize, Qt, QThread, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QMovie, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from ..models import MediaItem
from ..services.library import LibraryService, MediaPage


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


def _resample_lanczos():
    return Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def _qimage_from_pil(image: Image.Image) -> QImage:
    converted = image.convert("RGBA")
    raw = converted.tobytes("raw", "RGBA")
    qimage = QImage(
        raw,
        converted.width,
        converted.height,
        converted.width * 4,
        QImage.Format_RGBA8888,
    )
    return qimage.copy()


@dataclass(slots=True)
class _TimelineHeader:
    title: str
    top: int
    height: int


@dataclass(slots=True)
class _TimelineTile:
    item_id: str
    index: int
    rect: QRect


class PreviewLoadWorker(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, request_id: int, mode: str, source_path: str, target_size: QSize) -> None:
        super().__init__()
        self.request_id = request_id
        self.mode = mode
        self.source_path = source_path
        self.target_size = target_size

    def run(self) -> None:
        try:
            if self.mode == "image":
                payload = self._load_image_payload()
            elif self.mode == "video_frame":
                payload = self._load_video_frame_payload()
            else:
                raise ValueError(f"Unsupported preview load mode: {self.mode}")
            self.finished.emit(self.request_id, payload)
        except Exception as exc:
            self.failed.emit(self.request_id, str(exc))

    def _load_image_payload(self) -> dict[str, object]:
        source = Path(self.source_path)
        if not source.exists():
            raise FileNotFoundError(str(source))
        with Image.open(source) as image:
            prepared = ImageOps.exif_transpose(image)
            if prepared.mode not in {"RGB", "RGBA"}:
                prepared = prepared.convert("RGBA")
            prepared.thumbnail(
                (
                    max(1, self.target_size.width()),
                    max(1, self.target_size.height()),
                ),
                _resample_lanczos(),
            )
            return {"kind": "image", "image": _qimage_from_pil(prepared)}

    def _load_video_frame_payload(self) -> dict[str, object]:
        if cv2 is None:
            raise RuntimeError("OpenCV video preview support is unavailable.")
        capture = cv2.VideoCapture(self.source_path)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to read video preview: {self.source_path}")
        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Unable to decode video preview: {self.source_path}")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            image.thumbnail(
                (
                    max(1, self.target_size.width()),
                    max(1, self.target_size.height()),
                ),
                _resample_lanczos(),
            )
            return {"kind": "video_frame", "image": _qimage_from_pil(image)}
        finally:
            capture.release()


class _ThumbnailLoadSignals(QObject):
    completed = Signal(str, QImage, int, int)
    failed = Signal(str, int)


class _ThumbnailLoadTask(QRunnable):
    def __init__(self, item_id: str, thumbnail_path: str, tile_size: int, generation: int) -> None:
        super().__init__()
        self.item_id = item_id
        self.thumbnail_path = thumbnail_path
        self.tile_size = tile_size
        self.generation = generation
        self.signals = _ThumbnailLoadSignals()

    def run(self) -> None:
        image = QImage(self.thumbnail_path)
        if image.isNull():
            self.signals.failed.emit(self.item_id, self.generation)
            return
        scaled = image.scaled(
            self.tile_size,
            self.tile_size,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        source_x = max(0, (scaled.width() - self.tile_size) // 2)
        source_y = max(0, (scaled.height() - self.tile_size) // 2)
        cropped = scaled.copy(source_x, source_y, self.tile_size, self.tile_size)
        self.signals.completed.emit(self.item_id, cropped, self.tile_size, self.generation)


class TimelineViewport(QAbstractScrollArea):
    currentItemChanged = Signal(str)
    selectionChanged = Signal(list)
    requestMore = Signal()
    zoomChanged = Signal(int)

    MIN_ZOOM = 0
    MAX_ZOOM = 100
    MIN_TILE_SIZE = 96
    MAX_TILE_SIZE = 240
    TILE_CORNER_RADIUS = 14
    DEFAULT_CACHE_BUDGET_MB = 2048
    THUMBNAIL_BATCH_SIZE = 24
    PREFETCH_THUMBNAIL_BATCH_SIZE = 12

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[MediaItem] = []
        self._item_ids: list[str] = []
        self._item_index_by_id: dict[str, int] = {}
        self._headers: list[_TimelineHeader] = []
        self._tiles: list[_TimelineTile] = []
        self._tile_bottoms: list[int] = []
        self._selected_ids: set[str] = set()
        self._current_item_id = ""
        self._anchor_item_id = ""
        self._zoom_level = 58
        self._status_text = "No media items match the current view"
        self._has_more = False
        self._request_more_pending = False
        self._thumbnail_cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._thumbnail_cache_costs: dict[str, int] = {}
        self._thumbnail_cache_bytes = 0
        self._thumbnail_cache_budget_bytes = self.DEFAULT_CACHE_BUDGET_MB * 1024 * 1024
        self._thumbnail_queue: deque[str] = deque()
        self._thumbnail_prefetch_queue: deque[str] = deque()
        self._thumbnail_loading_ids: set[str] = set()
        self._queued_visible_thumbnail_ids: set[str] = set()
        self._queued_prefetch_thumbnail_ids: set[str] = set()
        self._full_thumbnail_prefetch_enabled = False
        self._prefetch_cursor = 0
        self._thumbnail_generation = 0
        self._thumbnail_pool = QThreadPool(self)
        self._thumbnail_pool.setMaxThreadCount(max(2, min(6, os.cpu_count() or 2)))
        self._last_interaction_at = 0.0
        self._placeholder_icon = self.style().standardIcon(QStyle.SP_FileIcon)
        self._thumbnail_timer = QTimer(self)
        self._thumbnail_timer.setSingleShot(True)
        self._thumbnail_timer.timeout.connect(self._load_next_thumbnail_batch)
        self.viewport().setMouseTracking(True)
        self.verticalScrollBar().valueChanged.connect(self._handle_viewport_changed)
        self.horizontalScrollBar().valueChanged.connect(self._handle_viewport_changed)

    def clear(self, status_text: str = "No media items match the current view") -> None:
        self._items = []
        self._item_ids = []
        self._item_index_by_id = {}
        self._headers = []
        self._tiles = []
        self._tile_bottoms = []
        self._selected_ids.clear()
        self._current_item_id = ""
        self._anchor_item_id = ""
        self._has_more = False
        self._request_more_pending = False
        self._thumbnail_queue = deque()
        self._thumbnail_prefetch_queue = deque()
        self._thumbnail_loading_ids.clear()
        self._queued_visible_thumbnail_ids.clear()
        self._queued_prefetch_thumbnail_ids.clear()
        self._prefetch_cursor = 0
        self._thumbnail_generation += 1
        self._status_text = status_text
        self.verticalScrollBar().setRange(0, 0)
        self.viewport().update()
        self.selectionChanged.emit([])
        self.currentItemChanged.emit("")

    def set_items(
        self,
        items: list[MediaItem],
        *,
        append: bool,
        has_more: bool,
        status_text: str,
    ) -> None:
        previous_current = self._current_item_id
        previous_selection = self.selected_item_ids()

        if append:
            existing_ids = set(self._item_ids)
            merged_items = list(self._items)
            for item in items:
                if item.id in existing_ids:
                    continue
                merged_items.append(item)
                existing_ids.add(item.id)
            self._items = merged_items
        else:
            self._items = list(items)
            self._thumbnail_queue = deque()
            self._thumbnail_prefetch_queue = deque()
            self._thumbnail_loading_ids.clear()
            self._queued_visible_thumbnail_ids.clear()
            self._queued_prefetch_thumbnail_ids.clear()
            self._prefetch_cursor = 0
            self._thumbnail_generation += 1

        self._item_ids = [item.id for item in self._items]
        self._item_index_by_id = {
            item_id: index
            for index, item_id in enumerate(self._item_ids)
        }
        self._has_more = has_more
        self._request_more_pending = False
        self._status_text = status_text

        retained_selection = {
            item_id
            for item_id in previous_selection
            if item_id in self._item_index_by_id
        }
        self._selected_ids = retained_selection
        if previous_current in self._item_index_by_id:
            self._current_item_id = previous_current
        elif self._items:
            self._current_item_id = self._items[0].id
            self._selected_ids = {self._current_item_id}
        else:
            self._current_item_id = ""
            self._selected_ids.clear()
        if self._current_item_id:
            self._anchor_item_id = self._current_item_id

        self._rebuild_layout()
        current_selection = self.selected_item_ids()
        if current_selection != previous_selection:
            self.selectionChanged.emit(current_selection)
        if self._current_item_id != previous_current:
            self.currentItemChanged.emit(self._current_item_id)

    def current_item_id(self) -> str:
        return self._current_item_id

    def selected_item_ids(self) -> list[str]:
        if not self._selected_ids:
            return []
        return [
            item_id
            for item_id in self._item_ids
            if item_id in self._selected_ids
        ]

    def zoom_level(self) -> int:
        return self._zoom_level

    def set_zoom_level(self, zoom_level: int) -> None:
        bounded = max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom_level))
        if bounded == self._zoom_level:
            return
        current_ratio = self._scroll_ratio()
        self._zoom_level = bounded
        self._last_interaction_at = monotonic()
        self._thumbnail_cache.clear()
        self._thumbnail_cache_costs.clear()
        self._thumbnail_cache_bytes = 0
        self._thumbnail_queue = deque()
        self._thumbnail_prefetch_queue = deque()
        self._thumbnail_loading_ids.clear()
        self._queued_visible_thumbnail_ids.clear()
        self._queued_prefetch_thumbnail_ids.clear()
        self._prefetch_cursor = 0
        self._thumbnail_generation += 1
        self._rebuild_layout(scroll_ratio=current_ratio)
        self.zoomChanged.emit(self._zoom_level)

    def is_interacting(self, *, idle_seconds: float = 1.25) -> bool:
        return monotonic() - self._last_interaction_at < idle_seconds

    def set_cache_budget_mb(self, budget_mb: int) -> None:
        bounded_mb = max(128, int(budget_mb))
        self._thumbnail_cache_budget_bytes = bounded_mb * 1024 * 1024
        self._enforce_thumbnail_cache_budget()

    def set_full_thumbnail_prefetch_enabled(self, enabled: bool) -> None:
        self._full_thumbnail_prefetch_enabled = enabled
        if enabled:
            self._queue_prefetch_thumbnails()
            if (self._thumbnail_queue or self._thumbnail_prefetch_queue) and not self._thumbnail_timer.isActive():
                self._thumbnail_timer.start(0)

    def header_granularity(self) -> str:
        if self._zoom_level <= 28:
            return "year"
        if self._zoom_level <= 62:
            return "month"
        return "day"

    def viewportEvent(self, event) -> bool:
        if event.type() == QEvent.Paint:
            self._paint_viewport()
            return True
        if event.type() == QEvent.MouseButtonPress:
            self._handle_mouse_press(event)
            return True
        if event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            step = 8 if delta > 0 else -8
            self.set_zoom_level(self._zoom_level + step)
            return True
        return super().viewportEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rebuild_layout(scroll_ratio=self._scroll_ratio())

    def _paint_viewport(self) -> None:
        painter = QPainter(self.viewport())
        painter.fillRect(self.viewport().rect(), QColor("#fbf9f5"))
        if not self._items:
            painter.setPen(QColor("#6d6458"))
            painter.drawText(
                self.viewport().rect().adjusted(24, 24, -24, -24),
                Qt.AlignCenter | Qt.TextWordWrap,
                self._status_text,
            )
            return

        scroll_y = self.verticalScrollBar().value()
        view_height = self.viewport().height()
        visible_top = scroll_y
        visible_bottom = scroll_y + view_height
        header_background = QColor("#f7f2ea")
        header_text = QColor("#54473b")
        for header in self._headers:
            if header.top + header.height < visible_top or header.top > visible_bottom:
                if header.top > visible_bottom:
                    break
                continue
            rect = QRect(0, header.top - scroll_y, self.viewport().width(), header.height)
            painter.fillRect(rect, header_background)
            painter.setPen(header_text)
            font = painter.font()
            font.setBold(True)
            font.setPointSizeF(max(10.0, self._tile_size() / 12.0))
            painter.setFont(font)
            painter.drawText(rect.adjusted(20, 0, -20, 0), Qt.AlignVCenter | Qt.AlignLeft, header.title)

        for tile in self._tiles_from_y(visible_top):
            if tile.rect.top() > visible_bottom:
                break
            draw_rect = QRect(tile.rect)
            draw_rect.translate(0, -scroll_y)
            item = self._items[tile.index]
            is_current = item.id == self._current_item_id
            is_selected = item.id in self._selected_ids
            border_color = QColor("#0d3b66") if is_current else QColor("#d8d1c6")
            fill_color = QColor("#e7e1d7") if is_selected else QColor("#f3eee6")
            painter.setRenderHint(QPainter.Antialiasing, True)
            path = QPainterPath()
            path.addRoundedRect(draw_rect, self.TILE_CORNER_RADIUS, self.TILE_CORNER_RADIUS)
            painter.fillPath(path, fill_color)
            painter.setPen(border_color)
            painter.drawPath(path)

            pixmap = self._thumbnail_cache.get(item.id)
            if pixmap is None or pixmap.isNull():
                self._draw_placeholder_tile(painter, draw_rect)
                continue

            painter.setClipPath(path)
            painter.drawPixmap(draw_rect, pixmap)
            painter.setClipping(False)

    def _draw_placeholder_tile(self, painter: QPainter, draw_rect: QRect) -> None:
        painter.fillRect(draw_rect.adjusted(2, 2, -2, -2), QColor("#ece6db"))
        placeholder = self._placeholder_icon.pixmap(min(draw_rect.width() // 2, 64), min(draw_rect.height() // 2, 64))
        x = draw_rect.center().x() - placeholder.width() // 2
        y = draw_rect.center().y() - placeholder.height() // 2
        painter.drawPixmap(x, y, placeholder)

    def _handle_mouse_press(self, event) -> None:
        hit_item_id = self._item_id_at(event.position().toPoint())
        if not hit_item_id:
            return
        self._last_interaction_at = monotonic()
        modifiers = event.modifiers()
        if modifiers & Qt.ShiftModifier and self._anchor_item_id in self._item_index_by_id:
            self._select_range(hit_item_id)
        elif modifiers & Qt.ControlModifier:
            if hit_item_id in self._selected_ids:
                self._selected_ids.remove(hit_item_id)
            else:
                self._selected_ids.add(hit_item_id)
            self._current_item_id = hit_item_id
            self._anchor_item_id = hit_item_id
        else:
            self._selected_ids = {hit_item_id}
            self._current_item_id = hit_item_id
            self._anchor_item_id = hit_item_id
        self._ensure_current_visible()
        self.viewport().update()
        self.selectionChanged.emit(self.selected_item_ids())
        self.currentItemChanged.emit(self._current_item_id)

    def _select_range(self, hit_item_id: str) -> None:
        anchor_index = self._item_index_by_id.get(self._anchor_item_id, 0)
        hit_index = self._item_index_by_id.get(hit_item_id, anchor_index)
        start_index = min(anchor_index, hit_index)
        end_index = max(anchor_index, hit_index)
        self._selected_ids = set(self._item_ids[start_index : end_index + 1])
        self._current_item_id = hit_item_id

    def _item_id_at(self, position: QPoint) -> str:
        scroll_y = self.verticalScrollBar().value()
        absolute = QPoint(position.x(), position.y() + scroll_y)
        for tile in self._tiles_from_y(absolute.y()):
            if tile.rect.top() > absolute.y():
                break
            if tile.rect.contains(absolute):
                return tile.item_id
        return ""

    def _rebuild_layout(self, *, scroll_ratio: float | None = None) -> None:
        if not self._items:
            self._headers = []
            self._tiles = []
            self._tile_bottoms = []
            self.verticalScrollBar().setRange(0, 0)
            self.viewport().update()
            return

        tile_size = self._tile_size()
        spacing = max(10, tile_size // 12)
        top_margin = max(12, tile_size // 10)
        left_margin = max(12, tile_size // 10)
        header_height = max(36, tile_size // 3)
        section_gap = max(14, spacing)
        available_width = max(1, self.viewport().width() - (left_margin * 2))
        columns = max(1, (available_width + spacing) // (tile_size + spacing))
        total_row_width = columns * tile_size + (columns - 1) * spacing
        x_offset = max(left_margin, (self.viewport().width() - total_row_width) // 2)

        headers: list[_TimelineHeader] = []
        tiles: list[_TimelineTile] = []
        y = top_margin
        grouped_items = self._group_item_indexes(columns)
        for title, indexes in grouped_items:
            headers.append(_TimelineHeader(title=title, top=y, height=header_height))
            y += header_height + spacing
            for start in range(0, len(indexes), columns):
                row_indexes = indexes[start : start + columns]
                for column, item_index in enumerate(row_indexes):
                    x = x_offset + column * (tile_size + spacing)
                    rect = QRect(x, y, tile_size, tile_size)
                    tiles.append(
                        _TimelineTile(
                            item_id=self._items[item_index].id,
                            index=item_index,
                            rect=rect,
                        )
                    )
                y += tile_size + spacing
            y += section_gap

        content_height = max(0, y)
        self._headers = headers
        self._tiles = tiles
        self._tile_bottoms = [tile.rect.bottom() for tile in tiles]
        scrollbar = self.verticalScrollBar()
        maximum = max(0, content_height - self.viewport().height())
        scrollbar.setRange(0, maximum)
        scrollbar.setPageStep(self.viewport().height())
        if scroll_ratio is not None and maximum > 0:
            scrollbar.setValue(int(maximum * scroll_ratio))
        self._schedule_visible_asset_work()
        self.viewport().update()

    def _group_item_indexes(self, _columns: int) -> list[tuple[str, list[int]]]:
        groups: list[tuple[str, list[int]]] = []
        current_title = ""
        current_indexes: list[int] = []
        for index, item in enumerate(self._items):
            title = self._header_title(item)
            if current_title and title != current_title:
                groups.append((current_title, current_indexes))
                current_indexes = []
            current_title = title
            current_indexes.append(index)
        if current_indexes:
            groups.append((current_title, current_indexes))
        return groups

    def _header_title(self, item: MediaItem) -> str:
        captured = self._item_datetime(item)
        granularity = self.header_granularity()
        if granularity == "year":
            return captured.strftime("%Y")
        if granularity == "month":
            return captured.strftime("%B %Y")
        return captured.strftime("%B %d, %Y")

    def _item_datetime(self, item: MediaItem) -> datetime:
        try:
            return datetime.fromisoformat(item.captured_at.replace("Z", "+00:00"))
        except Exception:
            return datetime.fromtimestamp(max(item.modified_ts, 0))

    def _tile_size(self) -> int:
        ratio = (self._zoom_level - self.MIN_ZOOM) / max(1, self.MAX_ZOOM - self.MIN_ZOOM)
        return int(self.MIN_TILE_SIZE + ratio * (self.MAX_TILE_SIZE - self.MIN_TILE_SIZE))

    def _schedule_visible_asset_work(self) -> None:
        self._queue_visible_thumbnails()
        if self._full_thumbnail_prefetch_enabled:
            self._queue_prefetch_thumbnails()
        self._maybe_request_more()

    def _handle_viewport_changed(self, *_args) -> None:
        self._last_interaction_at = monotonic()
        self._schedule_visible_asset_work()
        self.viewport().update()

    def _queue_visible_thumbnails(self) -> None:
        if not self._items:
            return
        scroll_y = self.verticalScrollBar().value()
        overscan = self._tile_size() * 2
        visible_top = max(0, scroll_y - overscan)
        visible_bottom = scroll_y + self.viewport().height() + overscan
        for tile in self._tiles_from_y(visible_top):
            if tile.rect.top() > visible_bottom:
                break
            if tile.item_id in self._thumbnail_cache or tile.item_id in self._queued_visible_thumbnail_ids:
                continue
            if tile.item_id in self._thumbnail_loading_ids:
                continue
            item = self._items[tile.index]
            if not item.thumbnail_path or not Path(item.thumbnail_path).exists():
                continue
            self._thumbnail_queue.append(tile.item_id)
            self._queued_visible_thumbnail_ids.add(tile.item_id)
        if (self._thumbnail_queue or self._thumbnail_prefetch_queue) and not self._thumbnail_timer.isActive():
            self._thumbnail_timer.start(0)

    def _queue_prefetch_thumbnails(self, *, max_items: int = 512) -> None:
        if not self._items:
            return
        queued = 0
        while self._prefetch_cursor < len(self._items) and queued < max_items:
            item = self._items[self._prefetch_cursor]
            self._prefetch_cursor += 1
            if item.id in self._thumbnail_cache or item.id in self._queued_prefetch_thumbnail_ids:
                continue
            if item.id in self._thumbnail_loading_ids:
                continue
            if not item.thumbnail_path or not Path(item.thumbnail_path).exists():
                continue
            self._thumbnail_prefetch_queue.append(item.id)
            self._queued_prefetch_thumbnail_ids.add(item.id)
            queued += 1

    def _load_next_thumbnail_batch(self) -> None:
        if not self._thumbnail_queue and self._thumbnail_prefetch_queue and self.is_interacting(idle_seconds=0.35):
            self._thumbnail_timer.start(35)
            return

        available_slots = max(0, self._thumbnail_pool.maxThreadCount() - self._thumbnail_pool.activeThreadCount())
        if available_slots <= 0:
            self._thumbnail_timer.start(12)
            return

        batch_size = min(
            available_slots,
            self.THUMBNAIL_BATCH_SIZE if self._thumbnail_queue else self.PREFETCH_THUMBNAIL_BATCH_SIZE,
        )
        for _ in range(batch_size):
            if self._thumbnail_queue:
                item_id = self._thumbnail_queue.popleft()
                self._queued_visible_thumbnail_ids.discard(item_id)
            elif self._thumbnail_prefetch_queue:
                item_id = self._thumbnail_prefetch_queue.popleft()
                self._queued_prefetch_thumbnail_ids.discard(item_id)
            else:
                break
            self._start_thumbnail_load_for_item_id(item_id)
        if self._full_thumbnail_prefetch_enabled and len(self._thumbnail_prefetch_queue) < self.PREFETCH_THUMBNAIL_BATCH_SIZE:
            self._queue_prefetch_thumbnails()
        if self._thumbnail_queue or self._thumbnail_prefetch_queue or self._thumbnail_pool.activeThreadCount() > 0:
            self._thumbnail_timer.start(8 if self._thumbnail_queue else 16)

    def _start_thumbnail_load_for_item_id(self, item_id: str) -> bool:
        if item_id in self._thumbnail_cache:
            self._thumbnail_cache.move_to_end(item_id)
            return False
        if item_id in self._thumbnail_loading_ids:
            return False
        item_index = self._item_index_by_id.get(item_id)
        if item_index is None:
            return False
        item = self._items[item_index]
        thumbnail_path = Path(item.thumbnail_path)
        if not item.thumbnail_path or not thumbnail_path.exists():
            return False
        self._thumbnail_loading_ids.add(item_id)
        task = _ThumbnailLoadTask(
            item_id,
            str(thumbnail_path),
            self._tile_size(),
            self._thumbnail_generation,
        )
        task.signals.completed.connect(self._handle_thumbnail_loaded)
        task.signals.failed.connect(self._handle_thumbnail_failed)
        self._thumbnail_pool.start(task)
        return True

    def _handle_thumbnail_loaded(self, item_id: str, image: QImage, tile_size: int, generation: int) -> None:
        self._thumbnail_loading_ids.discard(item_id)
        if generation != self._thumbnail_generation or tile_size != self._tile_size():
            return
        if item_id not in self._item_index_by_id or image.isNull():
            return
        self._store_thumbnail_pixmap(item_id, QPixmap.fromImage(image))
        self.viewport().update()
        if self._thumbnail_queue or self._thumbnail_prefetch_queue:
            self._thumbnail_timer.start(0)

    def _handle_thumbnail_failed(self, item_id: str, generation: int) -> None:
        if generation == self._thumbnail_generation:
            self._thumbnail_loading_ids.discard(item_id)
        if self._thumbnail_queue or self._thumbnail_prefetch_queue:
            self._thumbnail_timer.start(16)

    def _store_thumbnail_pixmap(self, item_id: str, pixmap: QPixmap) -> None:
        previous_cost = self._thumbnail_cache_costs.pop(item_id, 0)
        if previous_cost:
            self._thumbnail_cache_bytes = max(0, self._thumbnail_cache_bytes - previous_cost)
        self._thumbnail_cache[item_id] = pixmap
        self._thumbnail_cache.move_to_end(item_id)
        cost = self._pixmap_cost_bytes(pixmap)
        self._thumbnail_cache_costs[item_id] = cost
        self._thumbnail_cache_bytes += cost
        self._enforce_thumbnail_cache_budget()

    def _enforce_thumbnail_cache_budget(self) -> None:
        while self._thumbnail_cache and self._thumbnail_cache_bytes > self._thumbnail_cache_budget_bytes:
            removed_item_id, _pixmap = self._thumbnail_cache.popitem(last=False)
            removed_cost = self._thumbnail_cache_costs.pop(removed_item_id, 0)
            self._thumbnail_cache_bytes = max(0, self._thumbnail_cache_bytes - removed_cost)

    def _pixmap_cost_bytes(self, pixmap: QPixmap) -> int:
        depth_bytes = max(4, pixmap.depth() // 8)
        return max(1, pixmap.width()) * max(1, pixmap.height()) * depth_bytes

    def _tiles_from_y(self, y: int) -> Iterator[_TimelineTile]:
        start = bisect_left(self._tile_bottoms, y)
        for index in range(start, len(self._tiles)):
            yield self._tiles[index]

    def _maybe_request_more(self) -> None:
        if not self._has_more or self._request_more_pending:
            return
        scrollbar = self.verticalScrollBar()
        remaining = scrollbar.maximum() - scrollbar.value()
        if remaining > max(self.viewport().height(), self._tile_size() * 3):
            return
        self._request_more_pending = True
        self.requestMore.emit()

    def acknowledge_more_items(self) -> None:
        self._request_more_pending = False

    def _ensure_current_visible(self) -> None:
        if not self._current_item_id:
            return
        current_index = self._item_index_by_id.get(self._current_item_id)
        if current_index is None:
            return
        for tile in self._tiles:
            if tile.index != current_index:
                continue
            scrollbar = self.verticalScrollBar()
            top = scrollbar.value()
            bottom = top + self.viewport().height()
            if tile.rect.top() < top:
                scrollbar.setValue(tile.rect.top())
            elif tile.rect.bottom() > bottom:
                scrollbar.setValue(tile.rect.bottom() - self.viewport().height())
            return

    def _scroll_ratio(self) -> float:
        scrollbar = self.verticalScrollBar()
        if scrollbar.maximum() <= 0:
            return 0.0
        return scrollbar.value() / scrollbar.maximum()


class MediaGridWidget(QWidget):
    PAGE_SIZE = 180

    selectionChanged = Signal(list)

    def __init__(self, service: LibraryService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self._page_loader: Callable[[int, int], MediaPage] | None = None
        self._static_items: list[MediaItem] | None = None
        self._next_offset: int | None = 0
        self._has_more = False
        self._loading_page = False
        self._empty_message = "No media items match the current view"
        self._movie: QMovie | None = None
        self._preview_video_path = ""
        self._preview_request_id = 0
        self._preview_thread: QThread | None = None
        self._preview_worker: PreviewLoadWorker | None = None
        self._pending_preview_request: tuple[int, str, str] | None = None
        self._prefetch_all_pages = bool(self.service.config.gallery_prefetch_all_thumbnails)
        self._page_prefetch_delay_ms = max(25, int(self.service.config.gallery_prefetch_page_delay_ms))

        self.timeline_view = TimelineViewport()
        self.timeline_view.set_cache_budget_mb(self.service.config.gallery_thumbnail_cache_mb)
        self.timeline_view.set_full_thumbnail_prefetch_enabled(self._prefetch_all_pages)
        self.timeline_view.selectionChanged.connect(self.selectionChanged.emit)
        self.timeline_view.currentItemChanged.connect(self._handle_current_item_changed)
        self.timeline_view.requestMore.connect(self._load_next_page)
        self.timeline_view.zoomChanged.connect(self._sync_zoom_controls)

        self._page_prefetch_timer = QTimer(self)
        self._page_prefetch_timer.setSingleShot(True)
        self._page_prefetch_timer.timeout.connect(self._prefetch_next_page)

        self.zoom_out_button = QPushButton("−")
        self.zoom_out_button.clicked.connect(lambda: self.timeline_view.set_zoom_level(self.timeline_view.zoom_level() - 8))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(self.timeline_view.MIN_ZOOM, self.timeline_view.MAX_ZOOM)
        self.zoom_slider.setValue(self.timeline_view.zoom_level())
        self.zoom_slider.valueChanged.connect(self.timeline_view.set_zoom_level)
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.clicked.connect(lambda: self.timeline_view.set_zoom_level(self.timeline_view.zoom_level() + 8))

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom"))
        zoom_row.addWidget(self.zoom_out_button)
        zoom_row.addWidget(self.zoom_slider, 1)
        zoom_row.addWidget(self.zoom_in_button)

        gallery_panel = QWidget()
        gallery_layout = QVBoxLayout(gallery_panel)
        gallery_layout.addLayout(zoom_row)
        gallery_layout.addWidget(self.timeline_view, 1)

        self.preview_label = QLabel("Select a media item")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumWidth(420)
        self.preview_label.setMinimumHeight(420)
        self.preview_label.setObjectName("PreviewPanel")
        self.preview_label.setWordWrap(True)

        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(self.preview_label)

        self.video_status_label = QLabel("Video playback is unavailable in this environment.")
        self.video_status_label.setAlignment(Qt.AlignCenter)
        self.video_status_label.setMinimumWidth(420)
        self.video_status_label.setMinimumHeight(420)
        self.video_status_label.setObjectName("VideoPreviewPanel")
        self.video_status_label.setWordWrap(True)
        self.preview_stack.addWidget(self.video_status_label)

        self.video_widget = None
        self.media_player = None
        self.audio_output = None

        self.video_toggle_button = QPushButton("Play Preview")
        self.video_toggle_button.clicked.connect(self._toggle_video)
        self.video_toggle_button.setVisible(False)

        self.info_toggle_button = QPushButton("Show Info")
        self.info_toggle_button.clicked.connect(self._toggle_details)

        controls_row = QHBoxLayout()
        controls_row.addStretch(1)
        controls_row.addWidget(self.video_toggle_button)
        controls_row.addWidget(self.info_toggle_button)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumWidth(420)
        self.details.setVisible(False)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.addWidget(self.preview_stack)
        preview_layout.addLayout(controls_row)
        preview_layout.addWidget(self.details)

        splitter = QSplitter()
        splitter.addWidget(gallery_panel)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def set_page_loader(
        self,
        loader: Callable[[int, int], MediaPage],
        *,
        empty_message: str = "No media items match the current view",
    ) -> None:
        self._page_loader = loader
        self._static_items = None
        self._empty_message = empty_message
        self._next_offset = 0
        self._has_more = False
        self._loading_page = False
        self._page_prefetch_timer.stop()
        self.timeline_view.clear(status_text="Loading media items...")
        self._load_next_page(reset=True)
        self._schedule_page_prefetch()

    def set_items(self, items: list[MediaItem]) -> None:
        self._static_items = list(items)
        self.set_page_loader(
            lambda offset, limit: self._page_from_static_items(offset, limit),
            empty_message="No media items match the current view",
        )

    def current_item_id(self) -> str:
        return self.timeline_view.current_item_id()

    def selected_item_ids(self) -> list[str]:
        return self.timeline_view.selected_item_ids()

    def _page_from_static_items(self, offset: int, limit: int) -> MediaPage:
        items = self._static_items or []
        visible = items[offset : offset + limit]
        next_offset = offset + len(visible)
        has_more = next_offset < len(items)
        return MediaPage(items=visible, has_more=has_more, next_offset=next_offset if has_more else None)

    def _load_next_page(self, *, reset: bool = False) -> None:
        if self._page_loader is None or self._loading_page:
            return
        offset = 0 if reset or self._next_offset is None else self._next_offset
        self._loading_page = True
        try:
            page = self._page_loader(offset, self.PAGE_SIZE)
        finally:
            self._loading_page = False
        self._next_offset = page.next_offset
        self._has_more = page.has_more
        self.timeline_view.set_items(
            page.items,
            append=not reset and offset > 0,
            has_more=page.has_more,
            status_text=self._empty_message,
        )
        self.timeline_view.acknowledge_more_items()
        self._schedule_page_prefetch()

    def is_interacting(self) -> bool:
        return self.timeline_view.is_interacting()

    def _schedule_page_prefetch(self) -> None:
        if not self._prefetch_all_pages or not self._has_more or self._page_loader is None:
            return
        if self._page_prefetch_timer.isActive():
            return
        self._page_prefetch_timer.start(self._page_prefetch_delay_ms)

    def _prefetch_next_page(self) -> None:
        if not self._prefetch_all_pages or not self._has_more or self._page_loader is None:
            return
        if self._loading_page:
            self._schedule_page_prefetch()
            return
        if self.timeline_view.is_interacting(idle_seconds=0.4):
            self._page_prefetch_timer.start(self._page_prefetch_delay_ms)
            return
        self._load_next_page()

    def _sync_zoom_controls(self, zoom_level: int) -> None:
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(zoom_level)
        self.zoom_slider.blockSignals(False)

    def _handle_current_item_changed(self, item_id: str) -> None:
        self._clear_preview_media()
        item = self._item_for_id(item_id)
        if item is None:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Select a media item")
            self.preview_stack.setCurrentWidget(self.preview_label)
            self.video_toggle_button.setVisible(False)
            self.details.setText("")
            return

        self.details.setText(self.service.build_item_details(item))
        if item.media_kind == "gif":
            self._show_gif_preview(item)
            return
        if item.media_kind in {"video", "live_photo"}:
            self._show_motion_preview(item)
            return
        self._show_image_preview(item)

    def _item_for_id(self, item_id: str) -> MediaItem | None:
        if not item_id:
            return None
        if self._static_items is not None:
            for item in self._static_items:
                if item.id == item_id:
                    return item
        for item in self.timeline_view._items:
            if item.id == item_id:
                return item
        return None

    def _show_image_preview(self, item: MediaItem) -> None:
        image_source = self._image_source_for_item(item)
        if not image_source:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(item.title)
            self.preview_stack.setCurrentWidget(self.preview_label)
            self.video_toggle_button.setVisible(False)
            return
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("Loading full-resolution preview…")
        self.preview_stack.setCurrentWidget(self.preview_label)
        self.video_toggle_button.setVisible(False)
        self._queue_preview_request("image", image_source)

    def _show_gif_preview(self, item: MediaItem) -> None:
        if Path(item.path).exists():
            movie = QMovie(item.path)
            if movie.isValid():
                movie.setScaledSize(self._preview_target_size())
                self.preview_label.setMovie(movie)
                movie.start()
                self._movie = movie
                self.preview_label.setText("")
                self.preview_stack.setCurrentWidget(self.preview_label)
                self.video_toggle_button.setVisible(False)
                return
        self._show_image_preview(item)

    def _show_motion_preview(self, item: MediaItem) -> None:
        self._preview_video_path = self._video_path_for_item(item)
        image_source = self._image_source_for_item(item)
        if image_source:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Loading full-resolution preview…")
            self.preview_stack.setCurrentWidget(self.preview_label)
            self._queue_preview_request("image", image_source)
        elif self._preview_video_path:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Loading video preview…")
            self.preview_stack.setCurrentWidget(self.preview_label)
            self._queue_preview_request("video_frame", self._preview_video_path)
        else:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(item.title)
            self.preview_stack.setCurrentWidget(self.preview_label)
        self.video_toggle_button.setVisible(bool(self._preview_video_path))
        if self._preview_video_path:
            self.video_toggle_button.setText("Play Preview")

    def _image_source_for_item(self, item: MediaItem) -> str:
        if Path(item.path).suffix.lower() in IMAGE_EXTENSIONS and Path(item.path).exists():
            return item.path
        for component in item.component_paths:
            component_path = Path(component)
            if component_path.suffix.lower() in IMAGE_EXTENSIONS and component_path.exists():
                return component
        return ""

    def _clear_preview_media(self) -> None:
        self._preview_request_id += 1
        self._preview_video_path = ""
        if self._movie is not None:
            self._movie.stop()
            self._movie = None
        self.preview_label.clear()
        if self.media_player is not None:
            self.media_player.stop()
            self.media_player.setSource(QUrl())

    def _queue_preview_request(self, mode: str, source_path: str) -> None:
        self._preview_request_id += 1
        request = (self._preview_request_id, mode, source_path)
        if self._preview_thread is not None:
            self._pending_preview_request = request
            return
        self._start_preview_request(request)

    def _start_preview_request(self, request: tuple[int, str, str]) -> None:
        request_id, mode, source_path = request
        thread = QThread(self)
        worker = PreviewLoadWorker(request_id, mode, source_path, self._preview_target_size())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_preview_loaded)
        worker.failed.connect(self._handle_preview_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_preview_handles)

        self._preview_thread = thread
        self._preview_worker = worker
        thread.start()

    def _handle_preview_loaded(self, request_id: int, payload: object) -> None:
        if request_id != self._preview_request_id:
            return
        if not isinstance(payload, dict):
            return
        qimage = payload.get("image")
        if not isinstance(qimage, QImage):
            return
        pixmap = QPixmap.fromImage(qimage)
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        self.preview_label.setText("")
        self.preview_stack.setCurrentWidget(self.preview_label)

    def _handle_preview_failed(self, request_id: int, message: str) -> None:
        if request_id != self._preview_request_id:
            return
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText(f"Unable to load preview.\n\n{message}")
        self.preview_stack.setCurrentWidget(self.preview_label)

    def _clear_preview_handles(self) -> None:
        self._preview_thread = None
        self._preview_worker = None
        if self._pending_preview_request is not None:
            request = self._pending_preview_request
            self._pending_preview_request = None
            QTimer.singleShot(0, lambda request=request: self._start_preview_request(request))

    def _preview_target_size(self) -> QSize:
        size = self.preview_stack.size()
        return QSize(max(720, size.width()), max(720, size.height()))

    def _toggle_details(self) -> None:
        self.details.setVisible(not self.details.isVisible())
        self.info_toggle_button.setText("Hide Info" if self.details.isVisible() else "Show Info")

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
