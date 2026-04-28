from __future__ import annotations

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import difflib
from itertools import combinations
import math
import os
from pathlib import Path
import re
from threading import Lock
from time import monotonic
from typing import Callable, Iterable

try:
    import cv2
except Exception:
    cv2 = None

from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

try:
    import imagehash
except Exception:
    imagehash = None

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

from ..config import AppConfig
from ..media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, MediaAssetSpec, build_asset_specs, stable_id
from ..models import Album, DetectionRecord, DetectionRegion, LibraryState, MediaItem, Memory, Persona, utc_now
from ..store import SQLiteLibraryStore
from .model_manager import ModelManager, ModelStatus
from .vision import AnalysisResult, CAT_LABELS, DOG_LABELS, PET_LABELS, PreparedAssetInput, VideoFrameSample, VisionAnalyzer


PERSONA_COLORS = [
    "#0D3B66",
    "#3A7D44",
    "#A23B72",
    "#B56576",
    "#33658A",
    "#6D597A",
    "#5E6472",
]

REFERENCE_EMBEDDING_SIMILARITY_THRESHOLD = 0.985
REFERENCE_SIGNATURE_DISTANCE_THRESHOLD = 8.0
UNKNOWN_PERSON_CLUSTER_SIMILARITY_FLOOR = 0.68
UNKNOWN_PERSON_CLUSTER_SIMILARITY_BOOST = 0.12
UNKNOWN_PERSON_CLUSTER_SIMILARITY_CAP = 0.82
UNKNOWN_PET_CLUSTER_SIMILARITY_FLOOR = 0.78
UNKNOWN_PET_CLUSTER_SIMILARITY_BOOST = 0.08
UNKNOWN_PET_CLUSTER_SIMILARITY_CAP = 0.90
UNKNOWN_PERSON_CLUSTER_MERGE_MARGIN = 0.05
UNKNOWN_PET_CLUSTER_MERGE_MARGIN = 0.04
UNKNOWN_CLUSTER_SIGNATURE_MERGE_SLACK = 2.0
SCAN_PROGRESS_EMIT_INTERVAL = 48
SEARCH_CANDIDATE_LIMIT = 600
CLUSTER_DIRTY_CHUNK_SIZE = 512


@dataclass(slots=True)
class SyncSummary:
    added: int
    updated: int
    removed: int


@dataclass(slots=True)
class MediaPage:
    items: list[MediaItem]
    has_more: bool
    next_offset: int | None


@dataclass(slots=True)
class ProgressUpdate:
    phase: str
    message: str
    current: int = 0
    total: int = 0
    detail: str = ""
    indeterminate: bool = False
    snapshot_ready: bool = False
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    step_seconds: float | None = None
    timestamp_seconds: float | None = None


@dataclass(slots=True)
class PreparedSyncItem:
    spec: MediaAssetSpec
    existing: MediaItem | None
    analysis_mode: str
    metadata: dict[str, object]
    thumbnail_path: str
    still_image: Image.Image | None
    video_frames: list[VideoFrameSample]
    video_metadata: dict[str, object]


@dataclass(slots=True)
class UnknownClusterMember:
    item_id: str
    region_id: str
    label: str
    confidence: float
    captured_at: str


@dataclass(slots=True)
class UnknownPersonaCluster:
    id: str
    kind: str
    label: str
    member_count: int
    item_count: int
    member_ids: list[tuple[str, str]]
    item_ids: list[str]
    preview_path: str
    latest_captured_at: str
    average_confidence: float
    representative_item_id: str = ""
    representative_detection_id: str = ""
    revision: str = ""
    is_partial: bool = False
    updated_at: str = ""


@dataclass(slots=True)
class EmbeddingMapPoint:
    item_id: str
    detection_id: str
    label: str
    confidence: float
    captured_at: str
    embedding: list[float]
    persona_id: str = ""
    persona_name: str = ""
    persona_color: str = "#A0A0A0"
    persona_preview_path: str = ""
    cluster_id: str = ""
    cluster_label: str = ""
    cluster_preview_path: str = ""
    thumbnail_path: str = ""
    source_path: str = ""


@dataclass(slots=True)
class UnknownClusterPersonaSuggestion:
    persona: Persona
    score: float
    method: str


@dataclass(slots=True)
class _UnknownClusterState:
    cluster_id: str
    kind: str
    members: list[UnknownClusterMember] = field(default_factory=list)
    detection_records: list[DetectionRecord] = field(default_factory=list)
    item_ids: set[str] = field(default_factory=set)
    labels: Counter[str] = field(default_factory=Counter)
    embeddings: list[list[float]] = field(default_factory=list)
    signatures: list[str] = field(default_factory=list)
    representative_item_id: str = ""
    representative_region_id: str = ""
    representative_confidence: float = -1.0
    latest_captured_at: str = ""


class LibraryService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.store = SQLiteLibraryStore(config.database_file)
        self.state = LibraryState(
            schema_version=self.store.schema_version(),
            updated_at=self.store.updated_at(),
        )
        self._state_loaded = False
        self.model_manager = ModelManager(config)
        self._vision: VisionAnalyzer | None = None
        self._vision_lock = Lock()
        config.cache_path.mkdir(parents=True, exist_ok=True)

    @property
    def vision(self) -> VisionAnalyzer:
        if self._vision is None:
            with self._vision_lock:
                if self._vision is None:
                    self._vision = VisionAnalyzer(self.config, model_manager=self.model_manager)
        return self._vision

    def reload(self) -> None:
        self.state = LibraryState(
            schema_version=self.store.schema_version(),
            updated_at=self.store.updated_at(),
        )
        self._state_loaded = False
        self.model_manager = ModelManager(self.config)
        self._vision = None

    def _ensure_state_loaded(self) -> None:
        if self._state_loaded:
            return
        self.state = self.store.load()
        self._state_loaded = True
        if self._cleanup_collections():
            self.save()

    def sync(
        self,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
        *,
        include_pets: bool = True,
    ) -> SyncSummary:
        self._ensure_state_loaded()
        sync_started_at = monotonic()
        discovery_started_at = monotonic()
        self._emit_progress(
            progress_callback,
            self._make_progress_update(
                phase="sync",
                message="Discovering media files",
                detail=str(self.config.media_root_path),
                indeterminate=True,
                overall_started_at=sync_started_at,
                step_started_at=discovery_started_at,
            ),
        )

        last_discovery_emit_at = discovery_started_at

        def emit_discovery_progress(path_text: str, scanned_entries: int, discovered_media: int) -> None:
            nonlocal last_discovery_emit_at
            now = monotonic()
            if now - last_discovery_emit_at < 0.75:
                return
            last_discovery_emit_at = now
            self._emit_progress(
                progress_callback,
                self._make_progress_update(
                    phase="sync",
                    message="Discovering media files",
                    detail=f"{discovered_media} candidate files found after scanning {scanned_entries} paths — {path_text}",
                    indeterminate=True,
                    overall_started_at=sync_started_at,
                    step_started_at=discovery_started_at,
                ),
            )

        assets = build_asset_specs(
            self.config.media_root_path,
            progress_callback=emit_discovery_progress,
        )
        sorted_assets = self._sorted_asset_entries(assets)
        existing_ids = set(self.state.items)
        current_ids = set(assets)
        removed_ids = existing_ids - current_ids
        added = 0
        updated = 0
        total_work = len(removed_ids) + len(sorted_assets) + 2
        completed = 0
        changed_entries: list[tuple[str, MediaAssetSpec, MediaItem | None, str]] = []

        self._emit_progress(
            progress_callback,
            self._make_progress_update(
                phase="sync",
                message="Checking library for changes",
                current=completed,
                total=total_work,
                detail=f"{len(sorted_assets)} media items discovered in {self.config.media_root_path}",
                overall_started_at=sync_started_at,
                step_started_at=discovery_started_at,
            ),
        )

        for item_id in sorted(removed_ids):
            self.state.items.pop(item_id, None)
            completed += 1
            if self._should_emit_scan_progress(completed, total_work):
                self._emit_progress(
                    progress_callback,
                    self._make_progress_update(
                        phase="sync",
                        message="Removing missing items",
                        current=completed,
                        total=total_work,
                        detail=item_id,
                        overall_started_at=sync_started_at,
                    ),
                )
        if removed_ids:
            self._save_progress_items([], removed_item_ids=sorted(removed_ids))

        for item_id, spec in sorted_assets:
            existing = self.state.items.get(item_id)
            reanalysis_mode = self._reanalysis_mode(existing)
            needs_pet_analysis = (
                include_pets
                and existing is not None
                and existing.file_signature == spec.file_signature
                and not bool(existing.metadata.get("pet_analysis_enabled", False))
            )
            if (
                existing
                and existing.file_signature == spec.file_signature
                and reanalysis_mode == "none"
                and not needs_pet_analysis
            ):
                completed += 1
                if self._should_emit_scan_progress(completed, total_work):
                    self._emit_progress(
                        progress_callback,
                        self._make_progress_update(
                            phase="sync",
                            message="Scanning library",
                            current=completed,
                            total=total_work,
                            detail=spec.relative_key,
                            overall_started_at=sync_started_at,
                        ),
                    )
                continue

            analysis_mode = "full"
            if not include_pets:
                analysis_mode = "full_no_pets"
            if (
                existing
                and existing.file_signature == spec.file_signature
                and reanalysis_mode == "human_faces_only"
                and not needs_pet_analysis
            ):
                analysis_mode = "human_faces_only"
            changed_entries.append((item_id, spec, existing, analysis_mode))

        if changed_entries:
            self._emit_progress(
                progress_callback,
                self._make_progress_update(
                    phase="sync",
                    message="Initializing AI backends",
                    current=completed,
                    total=total_work,
                    detail="Preparing models for batched analysis",
                    indeterminate=True,
                    overall_started_at=sync_started_at,
                ),
            )
            _ = self.vision
            batch_size = self._scan_batch_size()
            max_workers = min(self._prefetch_workers(), max(1, batch_size), len(changed_entries))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                total_batches = (len(changed_entries) + batch_size - 1) // batch_size
                for batch_index, batch_start in enumerate(range(0, len(changed_entries), batch_size), start=1):
                    batch_entries = changed_entries[batch_start : batch_start + batch_size]
                    batch_started_at = monotonic()
                    self._emit_progress(
                        progress_callback,
                        self._make_progress_update(
                            phase="sync",
                            message=f"Prefetching batch {batch_index}/{total_batches}",
                            current=completed,
                            total=total_work,
                            detail=self._batch_detail(batch_entries),
                            indeterminate=True,
                            overall_started_at=sync_started_at,
                            step_started_at=batch_started_at,
                        ),
                    )
                    prepared_batch = self._prepare_batch(batch_entries, executor)
                    existing_by_id = {
                        prepared.spec.id: prepared.existing
                        for prepared in prepared_batch
                    }
                    self._emit_progress(
                        progress_callback,
                        self._make_progress_update(
                            phase="sync",
                            message=f"Running AI batch {batch_index}/{total_batches}",
                            current=completed,
                            total=total_work,
                            detail=f"{len(prepared_batch)} items",
                            indeterminate=True,
                            overall_started_at=sync_started_at,
                            step_started_at=batch_started_at,
                        ),
                    )
                    built_items = self._build_items_batch(prepared_batch)
                    for item in built_items:
                        self.state.items[item.id] = item
                        source_existing = existing_by_id.get(item.id)
                        if source_existing is None:
                            added += 1
                        else:
                            updated += 1
                    completed += len(built_items)
                    if built_items:
                        self._emit_progress(
                            progress_callback,
                            self._make_progress_update(
                                phase="sync",
                                message=f"Indexed batch {batch_index}/{total_batches}",
                                current=completed,
                                total=total_work,
                                detail=self._batch_detail(batch_entries),
                                overall_started_at=sync_started_at,
                                step_started_at=batch_started_at,
                            ),
                        )
                        self._save_progress_items(built_items)
                        self._emit_progress(
                            progress_callback,
                            self._make_progress_update(
                                phase="sync",
                                message=f"Updated live view for batch {batch_index}/{total_batches}",
                                current=completed,
                                total=total_work,
                                detail=f"{len(self.state.items)} indexed items",
                                snapshot_ready=True,
                                overall_started_at=sync_started_at,
                                step_started_at=batch_started_at,
                            ),
                        )

        completed += 1
        self._emit_progress(
            progress_callback,
            self._make_progress_update(
                phase="sync",
                message="Refreshing albums and memories",
                current=completed,
                total=total_work,
                overall_started_at=sync_started_at,
            ),
        )
        self._merge_current_store_assignments_into_state()
        self._cleanup_collections()
        self.regenerate_memories()
        self._save_memories_only()
        completed += 1
        self._emit_progress(
            progress_callback,
            self._make_progress_update(
                phase="sync",
                message="Library sync complete",
                current=completed,
                total=total_work,
                overall_started_at=sync_started_at,
            ),
        )
        return SyncSummary(added=added, updated=updated, removed=len(removed_ids))

    def save(self) -> None:
        self.state.updated_at = utc_now()
        self.store.save(self.state)

    def _save_progress_items(
        self,
        items: Iterable[MediaItem],
        *,
        removed_item_ids: Iterable[str] = (),
    ) -> None:
        buffered_items = list(items)
        self._merge_current_store_assignments_into_items(buffered_items)
        self.state.updated_at = utc_now()
        self.store.save_items_progress(
            buffered_items,
            removed_item_ids,
            updated_at=self.state.updated_at,
            schema_version=self.state.schema_version,
            personas=self.state.personas,
        )

    def _save_memories_only(self) -> None:
        self.state.updated_at = utc_now()
        self.store.save_memories(
            self.state.memories.values(),
            updated_at=self.state.updated_at,
            schema_version=self.state.schema_version,
        )

    def _merge_current_store_assignments_into_state(self) -> None:
        current_state = self.store.load()
        for persona in current_state.personas.values():
            self._merge_persona_into_state(persona)
        for item_id, item in self.state.items.items():
            current_item = current_state.items.get(item_id)
            if current_item is not None:
                self._merge_assignment_fields_from_item(item, current_item)

    def _merge_current_store_assignments_into_items(self, items: list[MediaItem]) -> None:
        if not items:
            return
        for persona in self.store.list_personas():
            self._merge_persona_into_state(persona)
        current_items = {
            item.id: item
            for item in self.store.load_items_by_ids([item.id for item in items])
        }
        for item in items:
            current_item = current_items.get(item.id)
            if current_item is not None:
                self._merge_assignment_fields_from_item(item, current_item)

    def _merge_assignment_fields_from_item(self, item: MediaItem, current_item: MediaItem) -> None:
        persona_ids = set(item.manual_persona_ids)
        persona_ids.update(
            persona_id
            for persona_id in current_item.manual_persona_ids
            if persona_id in self.state.personas
        )
        item.manual_persona_ids = sorted(persona_ids)
        self._merge_detection_assignments(item, current_item)

    def _merge_persona_into_state(self, persona: Persona) -> None:
        existing = self.state.personas.get(persona.id)
        if existing is None:
            self.state.personas[persona.id] = persona
            return
        if not existing.avatar_item_id and persona.avatar_item_id:
            existing.avatar_item_id = persona.avatar_item_id
        for encoding in persona.reference_encodings:
            if encoding not in existing.reference_encodings:
                existing.reference_encodings.append(encoding)
        existing_signatures = set(existing.reference_signatures)
        for signature in persona.reference_signatures:
            if signature and signature not in existing_signatures:
                existing.reference_signatures.append(signature)
                existing_signatures.add(signature)
        existing_reference_keys = {
            (
                reference.get("path", ""),
                reference.get("source_item_id", ""),
                reference.get("source_region_id", ""),
            )
            for reference in existing.reference_images
        }
        for reference in persona.reference_images:
            key = (
                reference.get("path", ""),
                reference.get("source_item_id", ""),
                reference.get("source_region_id", ""),
            )
            if key not in existing_reference_keys:
                existing.reference_images.append(reference)
                existing_reference_keys.add(key)

    def list_items(self) -> list[MediaItem]:
        if not self._state_loaded:
            return self.store.query_items()
        return sorted(
            self.state.items.values(),
            key=lambda item: (self._item_datetime(item), item.modified_ts),
            reverse=True,
        )

    def list_personas(self, kind: str = "all") -> list[Persona]:
        if not self._state_loaded:
            return self.store.list_personas(kind=kind)
        personas = list(self.state.personas.values())
        if kind != "all":
            personas = [persona for persona in personas if persona.kind == kind]
        return sorted(personas, key=lambda persona: (persona.kind, persona.name.lower()))

    def list_albums(self) -> list[Album]:
        if not self._state_loaded:
            return self.store.list_albums()
        return sorted(self.state.albums.values(), key=lambda album: album.name.lower())

    def list_memories(self) -> list[Memory]:
        if not self._state_loaded:
            return self.store.list_memories()
        return sorted(
            self.state.memories.values(),
            key=lambda memory: memory.end_date or memory.created_at,
            reverse=True,
        )

    def model_statuses(self) -> list[ModelStatus]:
        return self.model_manager.all_statuses()

    def missing_recommended_model_ids(self) -> list[str]:
        return [
            spec.id
            for spec in self.model_manager.recommended_specs()
            if not self.model_manager.status(spec.id).installed
        ]

    def download_recommended_models(
        self,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
    ) -> list[str]:
        specs = self.model_manager.recommended_specs()
        paths: list[str] = []
        total = len(specs)
        started_at = monotonic()
        for index, spec in enumerate(specs, start=1):
            installed = self.model_manager.status(spec.id).installed
            step_started_at = monotonic()
            self._emit_progress(
                progress_callback,
                self._make_progress_update(
                    phase="models",
                    message="Checking AI models" if installed else "Downloading AI models",
                    current=index - 1,
                    total=total,
                    detail=spec.title,
                    overall_started_at=started_at,
                    step_started_at=step_started_at,
                ),
            )
            paths.append(self.model_manager.download_model(spec.id))
            self._emit_progress(
                progress_callback,
                self._make_progress_update(
                    phase="models",
                    message="AI model ready",
                    current=index,
                    total=total,
                    detail=spec.title,
                    overall_started_at=started_at,
                    step_started_at=step_started_at,
                ),
            )
        self._vision = None
        return paths

    def download_model(
        self,
        model_id: str,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
    ) -> str:
        status = self.model_manager.status(model_id)
        started_at = monotonic()
        self._emit_progress(
            progress_callback,
            self._make_progress_update(
                phase="models",
                message="Checking AI models" if status.installed else "Downloading AI models",
                current=0,
                total=1,
                detail=status.title,
                overall_started_at=started_at,
                step_started_at=started_at,
            ),
        )
        path = self.model_manager.download_model(model_id)
        self._emit_progress(
            progress_callback,
            self._make_progress_update(
                phase="models",
                message="AI model ready" if status.installed else "Downloaded AI model",
                current=1,
                total=1,
                detail=status.title,
                overall_started_at=started_at,
                step_started_at=started_at,
            ),
        )
        self._vision = None
        return path

    def search_items(
        self,
        query: str = "",
        media_kind: str = "all",
        persona_kind: str = "all",
        persona_id: str = "",
        favorites_only: bool = False,
        limit: int | None = None,
    ) -> list[MediaItem]:
        if limit is not None:
            return self.search_items_page(
                query=query,
                media_kind=media_kind,
                persona_kind=persona_kind,
                persona_id=persona_id,
                favorites_only=favorites_only,
                offset=0,
                limit=limit,
            ).items
        if not self._state_loaded:
            return self._search_items_from_store(
                query=query,
                media_kind=media_kind,
                persona_kind=persona_kind,
                persona_id=persona_id,
                favorites_only=favorites_only,
                limit=limit,
            )
        items = [item for item in self.list_items() if not item.hidden]

        if media_kind != "all":
            items = [item for item in items if item.media_kind == media_kind]
        if favorites_only:
            items = [item for item in items if item.favorite]
        if persona_kind != "all":
            items = [
                item
                for item in items
                if any(
                    self.state.personas.get(persona_id_ref) is not None
                    and self.state.personas[persona_id_ref].kind == persona_kind
                    for persona_id_ref in self.item_persona_ids(item)
                )
            ]
        if persona_id:
            items = [item for item in items if persona_id in self.item_persona_ids(item)]

        field_filters, free_text = self._parse_query(query)
        items = self._apply_field_filters(items, field_filters)
        if not free_text:
            return items if limit is None else items[:limit]

        scored_items: list[tuple[int, MediaItem]] = []
        for item in items:
            score = self._score_item(item, free_text)
            if score >= 35:
                scored_items.append((score, item))
        scored_items.sort(
            key=lambda entry: (entry[0], self._item_datetime(entry[1]), entry[1].modified_ts),
            reverse=True,
        )
        ranked_items = [item for _, item in scored_items]
        return ranked_items

    def search_items_page(
        self,
        query: str = "",
        media_kind: str = "all",
        persona_kind: str = "all",
        persona_id: str = "",
        favorites_only: bool = False,
        *,
        offset: int = 0,
        limit: int = 180,
    ) -> MediaPage:
        return self._search_items_page_from_store(
            query=query,
            media_kind=media_kind,
            persona_kind=persona_kind,
            persona_id=persona_id,
            favorites_only=favorites_only,
            offset=offset,
            limit=limit,
        )

    def items_for_persona(self, persona_id: str, limit: int | None = None) -> list[MediaItem]:
        if limit is not None:
            return self.items_for_persona_page(persona_id, offset=0, limit=limit).items
        if not self._state_loaded:
            return self.store.query_items(persona_ids=[persona_id], limit=limit)
        items = [
            item
            for item in self.list_items()
            if persona_id in self.item_persona_ids(item)
        ]
        return items

    def items_for_persona_page(
        self,
        persona_id: str,
        *,
        offset: int = 0,
        limit: int = 180,
    ) -> MediaPage:
        if not persona_id:
            return MediaPage(items=[], has_more=False, next_offset=None)
        items = self.store.query_items(
            persona_ids=[persona_id],
            offset=offset,
            limit=limit + 1,
        )
        return self._page_from_items(items, offset=offset, limit=limit, pretrimmed=True)

    def items_for_album(self, album_id: str, limit: int | None = None) -> list[MediaItem]:
        if limit is not None:
            return self.items_for_album_page(album_id, offset=0, limit=limit).items
        if not self._state_loaded:
            album = self.store.load_album(album_id)
            if not album:
                return []
            item_ids = album.item_ids
            return self.store.load_items_by_ids(item_ids)
        album = self.state.albums.get(album_id)
        if not album:
            return []
        item_ids = album.item_ids
        return [self.state.items[item_id] for item_id in item_ids if item_id in self.state.items]

    def items_for_album_page(
        self,
        album_id: str,
        *,
        offset: int = 0,
        limit: int = 180,
    ) -> MediaPage:
        if not album_id:
            return MediaPage(items=[], has_more=False, next_offset=None)
        album = self.store.load_album(album_id)
        if not album:
            return MediaPage(items=[], has_more=False, next_offset=None)
        items = self.store.load_items_by_ids(album.item_ids[offset : offset + limit + 1])
        return self._page_from_items(items, offset=offset, limit=limit, pretrimmed=True)

    def items_for_memory(self, memory_id: str, limit: int | None = None) -> list[MediaItem]:
        if limit is not None:
            return self.items_for_memory_page(memory_id, offset=0, limit=limit).items
        if not self._state_loaded:
            memory = self.store.load_memory(memory_id)
            if not memory:
                return []
            item_ids = memory.item_ids
            return self.store.load_items_by_ids(item_ids)
        memory = self.state.memories.get(memory_id)
        if not memory:
            return []
        item_ids = memory.item_ids
        return [self.state.items[item_id] for item_id in item_ids if item_id in self.state.items]

    def items_for_memory_page(
        self,
        memory_id: str,
        *,
        offset: int = 0,
        limit: int = 180,
    ) -> MediaPage:
        if not memory_id:
            return MediaPage(items=[], has_more=False, next_offset=None)
        memory = self.store.load_memory(memory_id)
        if not memory:
            return MediaPage(items=[], has_more=False, next_offset=None)
        items = self.store.load_items_by_ids(memory.item_ids[offset : offset + limit + 1])
        return self._page_from_items(items, offset=offset, limit=limit, pretrimmed=True)

    def item_persona_ids(self, item: MediaItem) -> list[str]:
        persona_ids = set(item.manual_persona_ids)
        for detection in item.detections:
            if detection.persona_id:
                persona_ids.add(detection.persona_id)
        return sorted(persona_ids)

    def personas_for_item(self, item: MediaItem) -> list[Persona]:
        if not self._state_loaded:
            personas = self.store.load_personas_by_ids(self.item_persona_ids(item))
            return sorted(personas, key=lambda persona: persona.name.lower())
        personas = []
        for persona_id in self.item_persona_ids(item):
            persona = self.state.personas.get(persona_id)
            if persona:
                personas.append(persona)
        return sorted(personas, key=lambda persona: persona.name.lower())

    def persona_reference_images(self, persona_id: str) -> list[dict[str, str]]:
        persona = self.state.personas.get(persona_id) if self._state_loaded else self.store.load_persona(persona_id)
        if not persona:
            return []
        references = [
            entry
            for entry in persona.reference_images
            if entry.get("path") and Path(entry["path"]).exists()
        ]
        references.sort(key=lambda entry: entry.get("created_at", ""), reverse=True)
        return references

    def face_embedding_map_points(self, *, limit: int = 12000) -> list[EmbeddingMapPoint]:
        records = [
            record
            for record in self.store.query_detections(cluster_kind="person", dirty_only=False)
            if record.encoding
        ]
        if limit > 0:
            records = records[:limit]
        if not records:
            return []

        persona_ids = sorted({record.persona_id for record in records if record.persona_id})
        personas_by_id = {
            persona.id: persona
            for persona in self.store.load_personas_by_ids(persona_ids)
        }
        avatar_item_ids = [
            persona.avatar_item_id
            for persona in personas_by_id.values()
            if persona.avatar_item_id
        ]
        avatar_items_by_id = {
            item.id: item
            for item in self.store.load_items_by_ids(avatar_item_ids)
        }
        persona_preview_paths = {
            persona.id: self._persona_embedding_preview_path(
                persona,
                avatar_items_by_id.get(persona.avatar_item_id),
            )
            for persona in personas_by_id.values()
        }

        cluster_by_member: dict[tuple[str, str], dict[str, object]] = {}
        for cluster in self.store.list_unknown_clusters("person"):
            for member in cluster.get("member_ids", []):
                if isinstance(member, tuple) and len(member) == 2:
                    cluster_by_member[(str(member[0]), str(member[1]))] = cluster

        item_ids = sorted({record.item_id for record in records})
        items_by_id = {
            item.id: item
            for item in self.store.load_items_by_ids(item_ids)
        }

        points: list[EmbeddingMapPoint] = []
        for record in records:
            item = items_by_id.get(record.item_id)
            persona = personas_by_id.get(record.persona_id or "")
            persona_id = persona.id if persona else ""
            cluster = cluster_by_member.get((record.item_id, record.detection_id)) if not persona_id else None
            points.append(
                EmbeddingMapPoint(
                    item_id=record.item_id,
                    detection_id=record.detection_id,
                    label=record.label,
                    confidence=record.confidence,
                    captured_at=record.captured_at,
                    embedding=list(record.encoding),
                    persona_id=persona_id,
                    persona_name=persona.name if persona else "",
                    persona_color=persona.color if persona else "#A0A0A0",
                    persona_preview_path=persona_preview_paths.get(persona_id, "") if persona_id else "",
                    cluster_id=str(cluster.get("id", "")) if cluster else "",
                    cluster_label=str(cluster.get("label", "")) if cluster else "",
                    cluster_preview_path=str(cluster.get("preview_path", "")) if cluster else "",
                    thumbnail_path=item.thumbnail_path if item else "",
                    source_path=item.path if item else "",
                )
            )
        return points

    def _persona_embedding_preview_path(self, persona: Persona, avatar_item: MediaItem | None) -> str:
        for reference in sorted(
            persona.reference_images,
            key=lambda entry: entry.get("created_at", ""),
            reverse=True,
        ):
            path = reference.get("path", "")
            if path and Path(path).exists():
                return path
        if avatar_item and avatar_item.thumbnail_path and Path(avatar_item.thumbnail_path).exists():
            return avatar_item.thumbnail_path
        return ""

    def list_unknown_persona_clusters(
        self,
        kind: str = "person",
        *,
        allow_stale_cache: bool = False,
        build_if_missing: bool = True,
        include_pets: bool = False,
    ) -> list[UnknownPersonaCluster]:
        del allow_stale_cache
        requested_kinds = (
            ["person", "pet"]
            if kind == "all"
            else [kind]
        )
        clusters: list[UnknownPersonaCluster] = []
        for requested_kind in requested_kinds:
            if requested_kind not in {"person", "pet"}:
                continue
            self._ensure_state_loaded()
            persisted_clusters = [
                self._deserialize_unknown_cluster(entry)
                for entry in self.store.list_unknown_clusters(requested_kind)
            ]
            if persisted_clusters:
                clusters.extend(persisted_clusters)
                continue
            if requested_kind == "pet" and not include_pets:
                continue
            if not build_if_missing:
                continue
            self._refresh_unknown_clusters_kind(
                requested_kind,
                partial=False,
                progress_callback=None,
                overall_started_at=monotonic(),
                overall_current=0,
                overall_total=1,
            )
            clusters.extend(
                self._deserialize_unknown_cluster(entry)
                for entry in self.store.list_unknown_clusters(requested_kind)
            )
        return sorted(
            clusters,
            key=lambda cluster: (cluster.member_count, cluster.item_count, cluster.latest_captured_at),
            reverse=True,
        )

    def items_for_unknown_clusters(
        self,
        clusters: Iterable[UnknownPersonaCluster],
        limit: int | None = None,
    ) -> list[MediaItem]:
        if limit is not None:
            return self.items_for_unknown_clusters_page(clusters, offset=0, limit=limit).items
        item_ids: set[str] = set()
        for cluster in clusters:
            item_ids.update(cluster.item_ids)
        if not self._state_loaded:
            items = self.store.query_items_by_ids(sorted(item_ids), limit=limit)
        else:
            items = [self.state.items[item_id] for item_id in item_ids if item_id in self.state.items]
        ranked_items = sorted(
            items,
            key=lambda item: (self._item_datetime(item), item.modified_ts),
            reverse=True,
        )
        return ranked_items

    def items_for_unknown_clusters_page(
        self,
        clusters: Iterable[UnknownPersonaCluster],
        *,
        offset: int = 0,
        limit: int = 180,
    ) -> MediaPage:
        item_ids: set[str] = set()
        for cluster in clusters:
            item_ids.update(cluster.item_ids)
        if not item_ids:
            return MediaPage(items=[], has_more=False, next_offset=None)
        items = self.store.query_items_by_ids(
            sorted(item_ids),
            offset=offset,
            limit=limit + 1,
        )
        return self._page_from_items(items, offset=offset, limit=limit, pretrimmed=True)

    def unknown_cluster_persona_suggestions(
        self,
        cluster_id: str,
        *,
        limit: int = 6,
    ) -> list[UnknownClusterPersonaSuggestion]:
        cluster_state = self._unknown_cluster_state_by_id(cluster_id)
        if cluster_state is None:
            return []

        suggestions: list[UnknownClusterPersonaSuggestion] = []
        for persona in self.list_personas(kind=cluster_state.kind):
            score, method = self._score_unknown_cluster_for_persona(cluster_state, persona)
            if not method:
                continue
            suggestions.append(
                UnknownClusterPersonaSuggestion(
                    persona=persona,
                    score=score,
                    method=method,
                )
            )
        suggestions.sort(key=lambda entry: entry.score, reverse=True)
        return suggestions[: max(1, limit)]

    def build_item_details(self, item: MediaItem) -> str:
        personas = ", ".join(persona.name for persona in self.personas_for_item(item)) or "None"
        detected_subjects = self._detected_subject_summary(item)
        tags = ", ".join(item.tags[:24]) or "None"
        lines = [
            f"Title: {item.title}",
            f"Type: {item.media_kind}",
            f"Path: {item.path}",
            f"Captured: {item.captured_at}",
            f"Size: {item.width} x {item.height}",
            f"Duration: {item.duration_seconds:.1f}s" if item.duration_seconds else "Duration: n/a",
            f"Favorite: {'yes' if item.favorite else 'no'}",
            f"Assigned People/Pets: {personas}",
            f"Detected Subjects: {detected_subjects}",
            f"Tags: {tags}",
        ]
        video_ai_frames = item.metadata.get("video_ai_frames_analyzed")
        if video_ai_frames:
            lines.append(f"Video AI frames: {video_ai_frames}")
        human_face_detector_model = item.metadata.get("human_face_detector_model")
        human_face_recognizer_model = item.metadata.get("human_face_recognizer_model")
        human_face_device = item.metadata.get("human_face_device")
        human_face_detector_device = item.metadata.get("human_face_detector_device")
        human_face_recognizer_device = item.metadata.get("human_face_recognizer_device")
        human_face_backend_error = item.metadata.get("human_face_backend_error")
        object_device = item.metadata.get("object_device")
        pet_face_device = item.metadata.get("pet_face_device")
        pet_embedding_device = item.metadata.get("pet_embedding_device")
        if human_face_detector_model:
            lines.append(f"Human Face Detector: {human_face_detector_model}")
        if human_face_recognizer_model:
            lines.append(f"Human Face Recognizer: {human_face_recognizer_model}")
        if human_face_device:
            lines.append(f"Human Face Device: {human_face_device}")
        if human_face_detector_device:
            lines.append(f"Human Face Detector Device: {human_face_detector_device}")
        if human_face_recognizer_device:
            lines.append(f"Human Face Recognizer Device: {human_face_recognizer_device}")
        if object_device:
            lines.append(f"Object Device: {object_device}")
        if pet_face_device:
            lines.append(f"Pet Face Device: {pet_face_device}")
        if pet_embedding_device:
            lines.append(f"Pet Embedding Device: {pet_embedding_device}")
        if human_face_backend_error:
            lines.append(f"Human Face Backend Warning: {human_face_backend_error}")
        if item.component_paths and len(item.component_paths) > 1:
            lines.append(f"Components: {len(item.component_paths)}")
        return "\n".join(lines)

    def _detected_subject_summary(self, item: MediaItem) -> str:
        labels: list[str] = []
        for detection in item.detections:
            if detection.kind == "face":
                labels.append("human face")
                continue
            if self._is_pet_detection(detection):
                normalized = detection.label.lower()
                labels.append("pet" if normalized == "pet" else normalized)
        if not labels:
            return "None"

        counts = Counter(labels)
        ordered = sorted(
            counts.items(),
            key=lambda entry: (entry[1], entry[0]),
            reverse=True,
        )
        return ", ".join(
            f"{label} x{count}" if count > 1 else label
            for label, count in ordered
        )

    def _page_from_items(
        self,
        items: list[MediaItem],
        *,
        offset: int,
        limit: int,
        pretrimmed: bool = False,
    ) -> MediaPage:
        if limit <= 0:
            return MediaPage(items=[], has_more=False, next_offset=None)
        visible_items = items if pretrimmed else items[offset : offset + limit + 1]
        page_items = visible_items[:limit]
        has_more = len(visible_items) > limit
        next_offset = offset + len(page_items) if has_more else None
        return MediaPage(items=page_items, has_more=has_more, next_offset=next_offset)

    def rebuild_unknown_cluster_caches(
        self,
        *,
        partial: bool,
        include_pets: bool = False,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
    ) -> None:
        self._ensure_state_loaded()
        revision = self.state.updated_at
        started_at = monotonic()
        kinds = ("person", "pet") if include_pets else ("person",)
        total_kinds = len(kinds)
        for index, kind in enumerate(kinds, start=1):
            self._refresh_unknown_clusters_kind(
                kind,
                partial=partial,
                progress_callback=progress_callback,
                overall_started_at=started_at,
                overall_current=index - 1,
                overall_total=total_kinds,
            )

    def _serialize_unknown_cluster(self, cluster: UnknownPersonaCluster) -> dict[str, object]:
        return {
            "id": cluster.id,
            "kind": cluster.kind,
            "label": cluster.label,
            "member_count": cluster.member_count,
            "item_count": cluster.item_count,
            "representative_item_id": cluster.representative_item_id,
            "representative_detection_id": cluster.representative_detection_id,
            "member_ids": [
                [item_id, region_id]
                for item_id, region_id in cluster.member_ids
            ],
            "item_ids": list(cluster.item_ids),
            "preview_path": cluster.preview_path,
            "latest_captured_at": cluster.latest_captured_at,
            "average_confidence": cluster.average_confidence,
            "revision": cluster.revision,
            "is_partial": cluster.is_partial,
            "updated_at": cluster.updated_at,
        }

    def _deserialize_unknown_cluster(self, payload: dict[str, object]) -> UnknownPersonaCluster:
        return UnknownPersonaCluster(
            id=str(payload.get("id", "")),
            kind=str(payload.get("kind", "person")),
            label=str(payload.get("label", "")),
            member_count=int(payload.get("member_count", 0)),
            item_count=int(payload.get("item_count", 0)),
            representative_item_id=str(payload.get("representative_item_id", "")),
            representative_detection_id=str(payload.get("representative_detection_id", "")),
            member_ids=[
                (str(entry[0]), str(entry[1]))
                for entry in payload.get("member_ids", [])
                if isinstance(entry, (list, tuple)) and len(entry) == 2
            ],
            item_ids=[str(item_id) for item_id in payload.get("item_ids", [])],
            preview_path=str(payload.get("preview_path", "")),
            latest_captured_at=str(payload.get("latest_captured_at", "")),
            average_confidence=float(payload.get("average_confidence", 0.0)),
            revision=str(payload.get("revision", "")),
            is_partial=bool(payload.get("is_partial", False)),
            updated_at=str(payload.get("updated_at", "")),
        )

    def create_persona(self, name: str, kind: str) -> Persona:
        self._ensure_state_loaded()
        normalized = name.strip()
        if not normalized:
            raise ValueError("Persona name is required.")

        for persona in self.state.personas.values():
            if persona.name.lower() == normalized.lower() and persona.kind == kind:
                return persona

        identifier = stable_id(f"persona:{kind}:{normalized.lower()}:{utc_now()}")
        persona = Persona(
            id=identifier,
            name=normalized,
            kind=kind,
            created_at=utc_now(),
            color=PERSONA_COLORS[len(self.state.personas) % len(PERSONA_COLORS)],
        )
        self.state.personas[identifier] = persona
        self.save()
        return persona

    def assign_region_to_persona(
        self,
        item_id: str,
        region_id: str,
        persona_id: str = "",
        new_name: str = "",
        kind: str = "person",
    ) -> Persona:
        self._ensure_state_loaded()
        persona = self._resolve_persona(persona_id, new_name, kind)
        item = self.state.items[item_id]
        detection = next((entry for entry in item.detections if entry.id == region_id), None)
        if detection is None:
            raise ValueError("Selected detection no longer exists.")

        cluster = self._unknown_cluster_containing_detection(kind, item_id, region_id)
        member_ids = cluster.member_ids if cluster and cluster.member_ids else [(item_id, region_id)]
        updated_items = self._assign_detection_members_to_persona(
            persona,
            member_ids,
            allow_reassign=True,
        )
        if not updated_items:
            raise ValueError("No assignable detections were found.")
        self._persist_persona_detection_assignment(persona, updated_items)
        return persona

    def clear_region_assignment(self, item_id: str, region_id: str) -> None:
        self._ensure_state_loaded()
        item = self.state.items[item_id]
        for detection in item.detections:
            if detection.id == region_id:
                detection.persona_id = None
                break
        self.regenerate_memories()
        self.save()

    def assign_item_to_persona(
        self,
        item_id: str,
        persona_id: str = "",
        new_name: str = "",
        kind: str = "person",
    ) -> Persona:
        self._ensure_state_loaded()
        persona = self._resolve_persona(persona_id, new_name, kind)
        item = self.state.items[item_id]
        if persona.id not in item.manual_persona_ids:
            item.manual_persona_ids.append(persona.id)
        if persona.kind == "person":
            face_detections = [entry for entry in item.detections if entry.kind == "face"]
            if len(face_detections) == 1:
                self._remember_detection_reference(persona, item, face_detections[0])
        if persona.kind == "pet":
            for detection in item.detections:
                self._remember_detection_reference(persona, item, detection)
        if not persona.avatar_item_id:
            persona.avatar_item_id = item.id
        self.regenerate_memories()
        self.save()
        return persona

    def clear_item_personas(self, item_id: str) -> None:
        self._ensure_state_loaded()
        item = self.state.items[item_id]
        item.manual_persona_ids = []
        self.regenerate_memories()
        self.save()

    def _resolve_persona_for_live_assignment(
        self,
        persona_id: str,
        new_name: str,
        kind: str,
    ) -> Persona:
        if persona_id:
            persona = (
                self.state.personas.get(persona_id)
                if self._state_loaded
                else self.store.load_persona(persona_id)
            )
            if not persona:
                raise ValueError("Persona not found.")
            if persona.kind != kind:
                raise ValueError("Selected persona does not match the selected cluster type.")
            if self._state_loaded:
                self.state.personas[persona.id] = persona
            return persona

        normalized = new_name.strip()
        if not normalized:
            raise ValueError("Persona name is required.")
        for persona in self.list_personas(kind=kind):
            if persona.name.lower() == normalized.lower():
                if self._state_loaded:
                    self.state.personas[persona.id] = persona
                return persona

        persona_count = len(self.state.personas) if self._state_loaded else self.store.count_personas()
        persona = Persona(
            id=stable_id(f"persona:{kind}:{normalized.lower()}:{utc_now()}"),
            name=normalized,
            kind=kind,
            created_at=utc_now(),
            color=PERSONA_COLORS[persona_count % len(PERSONA_COLORS)],
        )
        if self._state_loaded:
            self.state.personas[persona.id] = persona
        return persona

    def assign_unknown_clusters_to_persona(
        self,
        clusters: Iterable[UnknownPersonaCluster],
        persona_id: str = "",
        new_name: str = "",
        kind: str = "person",
    ) -> Persona:
        cluster_list = list(clusters)
        if not cluster_list:
            raise ValueError("Select at least one cluster.")

        cluster_kinds = {cluster.kind for cluster in cluster_list}
        if len(cluster_kinds) > 1:
            raise ValueError("Select clusters of a single kind at a time.")
        resolved_kind = next(iter(cluster_kinds))
        if kind and kind != resolved_kind:
            raise ValueError("Selected clusters do not match the chosen persona kind.")

        persona = self._resolve_persona_for_live_assignment(persona_id, new_name, resolved_kind)
        cluster_list = self._refresh_unknown_clusters_from_store(cluster_list)
        member_ids = [
            (item_id, region_id)
            for cluster in cluster_list
            for item_id, region_id in cluster.member_ids
        ]
        updated_items = self._assign_detection_members_to_persona(
            persona,
            member_ids,
            allow_reassign=False,
        )
        if not updated_items:
            raise ValueError("The selected clusters no longer contain assignable detections.")

        self._persist_persona_detection_assignment(persona, updated_items)
        return persona

    def _assign_detection_members_to_persona(
        self,
        persona: Persona,
        member_ids: Iterable[tuple[str, str]],
        *,
        allow_reassign: bool = False,
    ) -> dict[str, MediaItem]:
        seen_regions: set[tuple[str, str]] = set()
        member_list: list[tuple[str, str]] = []
        for item_id, region_id in member_ids:
            key = (str(item_id), str(region_id))
            if not key[0] or not key[1] or key in seen_regions:
                continue
            seen_regions.add(key)
            member_list.append(key)

        item_ids = sorted({item_id for item_id, _ in member_list})
        items_by_id = {
            item.id: item
            for item in self.store.load_items_by_ids(item_ids)
        }
        updated_items: dict[str, MediaItem] = {}
        for item_id, region_id in member_list:
            item = items_by_id.get(item_id)
            if item is None:
                continue
            detection = next((entry for entry in item.detections if entry.id == region_id), None)
            if detection is None:
                continue
            if not self._persona_can_own_detection(persona, detection):
                continue
            if detection.persona_id and detection.persona_id != persona.id and not allow_reassign:
                continue
            detection.persona_id = persona.id
            self._remember_detection_reference(persona, item, detection)
            if not persona.avatar_item_id:
                persona.avatar_item_id = item.id
            updated_items[item.id] = item
        return updated_items

    def _persist_persona_detection_assignment(
        self,
        persona: Persona,
        updated_items: dict[str, MediaItem],
    ) -> None:
        updated_at = utc_now()
        self.store.save_persona_assignment(
            persona,
            updated_items.values(),
            updated_at=updated_at,
            schema_version=self.state.schema_version,
            personas=self.state.personas if self._state_loaded else {persona.id: persona},
        )
        if self._state_loaded:
            self.state.updated_at = updated_at
            self.state.personas[persona.id] = persona
            for item in updated_items.values():
                self.state.items[item.id] = item
            self.regenerate_memories()
            self._save_memories_only()

    def _persona_can_own_detection(self, persona: Persona, detection: DetectionRegion) -> bool:
        if persona.kind == "person":
            return detection.kind == "face"
        if persona.kind == "pet":
            return self._is_pet_detection(detection)
        return False

    def _unknown_cluster_containing_detection(
        self,
        kind: str,
        item_id: str,
        region_id: str,
    ) -> UnknownPersonaCluster | None:
        if kind not in {"person", "pet"}:
            return None
        for payload in self.store.list_unknown_clusters(kind):
            cluster = self._deserialize_unknown_cluster(payload)
            if (item_id, region_id) in cluster.member_ids:
                return cluster
        return None

    def _refresh_unknown_clusters_from_store(
        self,
        clusters: list[UnknownPersonaCluster],
    ) -> list[UnknownPersonaCluster]:
        by_id: dict[str, UnknownPersonaCluster] = {}
        for kind in sorted({cluster.kind for cluster in clusters if cluster.kind in {"person", "pet"}}):
            for payload in self.store.list_unknown_clusters(kind):
                cluster = self._deserialize_unknown_cluster(payload)
                by_id[cluster.id] = cluster
        return [by_id.get(cluster.id, cluster) for cluster in clusters]

    def create_album(self, name: str, item_ids: Iterable[str] = ()) -> Album:
        self._ensure_state_loaded()
        normalized = name.strip()
        if not normalized:
            raise ValueError("Album name is required.")
        album = Album(
            id=stable_id(f"album:{normalized.lower()}:{utc_now()}"),
            name=normalized,
            created_at=utc_now(),
            item_ids=[item_id for item_id in item_ids if item_id in self.state.items],
        )
        self.state.albums[album.id] = album
        self.save()
        return album

    def add_items_to_album(self, album_id: str, item_ids: Iterable[str]) -> None:
        self._ensure_state_loaded()
        album = self.state.albums[album_id]
        for item_id in item_ids:
            if item_id in self.state.items and item_id not in album.item_ids:
                album.item_ids.append(item_id)
        self.save()

    def delete_album(self, album_id: str) -> None:
        self._ensure_state_loaded()
        self.state.albums.pop(album_id, None)
        self.save()

    def toggle_favorite(self, item_ids: Iterable[str]) -> bool:
        self._ensure_state_loaded()
        selected_items = [self.state.items[item_id] for item_id in item_ids if item_id in self.state.items]
        if not selected_items:
            return False
        new_state = not all(item.favorite for item in selected_items)
        for item in selected_items:
            item.favorite = new_state
        self.save()
        return new_state

    def regenerate_memories(self) -> None:
        self._ensure_state_loaded()
        self.state.memories = {}
        if not self.config.auto_generate_memories:
            return

        items = self.list_items()
        candidates: list[tuple[int, Memory]] = []
        candidates.extend(self._build_month_memories(items))
        candidates.extend(self._build_day_memories(items))
        candidates.extend(self._build_span_memories(items))
        candidates.extend(self._build_persona_memories())
        candidates.extend(self._build_persona_pair_memories(items))
        candidates.extend(self._build_theme_memories(items))
        candidates.extend(self._build_favorite_memories(items))

        seen: set[str] = set()
        for _, memory in sorted(candidates, key=lambda entry: entry[0], reverse=True)[:24]:
            if memory.id in seen:
                continue
            seen.add(memory.id)
            self.state.memories[memory.id] = memory

    def _build_month_memories(self, items: list[MediaItem]) -> list[tuple[int, Memory]]:
        grouped_by_month: dict[str, list[MediaItem]] = {}
        for item in items:
            grouped_by_month.setdefault(self._item_datetime(item).strftime("%Y-%m"), []).append(item)

        memories: list[tuple[int, Memory]] = []
        for key, group in grouped_by_month.items():
            if len(group) < self.config.memory_min_items:
                continue
            month_date = datetime.strptime(key, "%Y-%m").replace(tzinfo=timezone.utc)
            memories.append(
                (
                    len(group) + 18,
                    Memory(
                        id=stable_id(f"memory:month:{key}"),
                        title=month_date.strftime("%B %Y"),
                        subtitle=f"{len(group)} moments",
                        summary=f"A monthly highlight reel built from {len(group)} media items.",
                        created_at=utc_now(),
                        memory_type="time",
                        item_ids=[item.id for item in group[:48]],
                        start_date=group[-1].captured_at,
                        end_date=group[0].captured_at,
                    ),
                )
            )
        return memories

    def _build_day_memories(self, items: list[MediaItem]) -> list[tuple[int, Memory]]:
        grouped_by_day: dict[str, list[MediaItem]] = {}
        for item in items:
            grouped_by_day.setdefault(self._item_datetime(item).strftime("%Y-%m-%d"), []).append(item)

        memories: list[tuple[int, Memory]] = []
        for key, group in grouped_by_day.items():
            if len(group) < self.config.memory_min_items + 2:
                continue
            day_date = datetime.strptime(key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            memories.append(
                (
                    len(group) + 24,
                    Memory(
                        id=stable_id(f"memory:day:{key}"),
                        title=day_date.strftime("%B %d, %Y"),
                        subtitle=f"{len(group)} moments from one day",
                        summary="A denser one-day story assembled from a single date cluster.",
                        created_at=utc_now(),
                        memory_type="day",
                        item_ids=[item.id for item in group[:48]],
                        start_date=group[-1].captured_at,
                        end_date=group[0].captured_at,
                    ),
                )
            )
        return memories

    def _build_span_memories(self, items: list[MediaItem]) -> list[tuple[int, Memory]]:
        if not items:
            return []

        chronological = list(reversed(items))
        windows: list[list[MediaItem]] = []
        current_window: list[MediaItem] = []
        previous_dt: datetime | None = None
        for item in chronological:
            current_dt = self._item_datetime(item)
            if previous_dt is not None and (current_dt.date() - previous_dt.date()).days > 2:
                if len(current_window) >= self.config.memory_min_items and len({self._item_datetime(entry).date() for entry in current_window}) >= 2:
                    windows.append(list(reversed(current_window)))
                current_window = []
            current_window.append(item)
            previous_dt = current_dt
        if len(current_window) >= self.config.memory_min_items and len({self._item_datetime(entry).date() for entry in current_window}) >= 2:
            windows.append(list(reversed(current_window)))

        memories: list[tuple[int, Memory]] = []
        for window in windows[:8]:
            start_dt = self._item_datetime(window[-1])
            end_dt = self._item_datetime(window[0])
            memories.append(
                (
                    len(window) + 26,
                    Memory(
                        id=stable_id(f"memory:span:{start_dt.date()}:{end_dt.date()}"),
                        title=f"{start_dt.strftime('%b %d')} to {end_dt.strftime('%b %d, %Y')}",
                        subtitle=f"{len(window)} moments across {len({self._item_datetime(item).date() for item in window})} days",
                        summary="A multi-day cluster that feels closer to a trip or weekend memory.",
                        created_at=utc_now(),
                        memory_type="span",
                        item_ids=[item.id for item in window[:48]],
                        start_date=window[-1].captured_at,
                        end_date=window[0].captured_at,
                    ),
                )
            )
        return memories

    def _build_persona_memories(self) -> list[tuple[int, Memory]]:
        memories: list[tuple[int, Memory]] = []
        for persona in self.list_personas():
            persona_items = self.items_for_persona(persona.id)
            if len(persona_items) < self.config.memory_min_items:
                continue
            memories.append(
                (
                    len(persona_items) + (30 if persona.kind == "pet" else 28),
                    Memory(
                        id=stable_id(f"memory:persona:{persona.id}"),
                        title=persona.name,
                        subtitle=f"{len(persona_items)} appearances",
                        summary=f"A memory reel centered on {persona.name}.",
                        created_at=utc_now(),
                        memory_type=f"{persona.kind}_persona",
                        item_ids=[item.id for item in persona_items[:48]],
                        persona_ids=[persona.id],
                        start_date=persona_items[-1].captured_at,
                        end_date=persona_items[0].captured_at,
                    ),
                )
            )
        return memories

    def _build_persona_pair_memories(self, items: list[MediaItem]) -> list[tuple[int, Memory]]:
        pair_counter: Counter[tuple[str, str]] = Counter()
        pair_items: dict[tuple[str, str], list[str]] = {}
        for item in items:
            persona_ids = self.item_persona_ids(item)
            if len(persona_ids) < 2:
                continue
            for pair in combinations(persona_ids[:4], 2):
                normalized_pair = tuple(sorted(pair))
                pair_counter[normalized_pair] += 1
                pair_items.setdefault(normalized_pair, []).append(item.id)

        memories: list[tuple[int, Memory]] = []
        for pair, count in pair_counter.most_common(6):
            if count < self.config.memory_min_items:
                continue
            left = self.state.personas.get(pair[0])
            right = self.state.personas.get(pair[1])
            if not left or not right:
                continue
            item_ids = pair_items[pair][:48]
            memories.append(
                (
                    count + 16,
                    Memory(
                        id=stable_id(f"memory:pair:{pair[0]}:{pair[1]}"),
                        title=f"{left.name} and {right.name}",
                        subtitle=f"{count} shared moments",
                        summary="A memory built from repeated appearances together.",
                        created_at=utc_now(),
                        memory_type="pair",
                        item_ids=item_ids,
                        persona_ids=list(pair),
                        start_date=self.state.items[item_ids[-1]].captured_at,
                        end_date=self.state.items[item_ids[0]].captured_at,
                    ),
                )
            )
        return memories

    def _build_theme_memories(self, items: list[MediaItem]) -> list[tuple[int, Memory]]:
        ignored_tags = {
            "image",
            "video",
            "gif",
            "live",
            "live photo",
            "live_photo",
            "face",
            "person",
            "pet",
            "animal",
        }
        themed_items: dict[str, list[MediaItem]] = {}
        for item in items:
            for tag in item.tags:
                normalized = tag.lower()
                if normalized in ignored_tags or len(normalized) < 3:
                    continue
                themed_items.setdefault(normalized, []).append(item)

        memories: list[tuple[int, Memory]] = []
        for tag, group in sorted(themed_items.items(), key=lambda entry: len(entry[1]), reverse=True)[:8]:
            if len(group) < self.config.memory_min_items:
                continue
            title = tag.title() if tag not in CAT_LABELS else "Cats"
            memories.append(
                (
                    len(group) + (14 if tag in CAT_LABELS else 8),
                    Memory(
                        id=stable_id(f"memory:theme:{tag}"),
                        title=title,
                        subtitle=f"{len(group)} tagged moments",
                        summary=f"A themed memory built around recurring {tag} moments.",
                        created_at=utc_now(),
                        memory_type="theme",
                        item_ids=[item.id for item in group[:48]],
                        start_date=group[-1].captured_at,
                        end_date=group[0].captured_at,
                    ),
                )
            )
        return memories

    def _build_favorite_memories(self, items: list[MediaItem]) -> list[tuple[int, Memory]]:
        favorite_items = [item for item in items if item.favorite]
        if len(favorite_items) < self.config.memory_min_items:
            return []
        return [
            (
                len(favorite_items) + 12,
                Memory(
                    id=stable_id("memory:favorites"),
                    title="Favorites",
                    subtitle=f"{len(favorite_items)} saved moments",
                    summary="A quick reel made from favorited items.",
                    created_at=utc_now(),
                    memory_type="favorites",
                    item_ids=[item.id for item in favorite_items[:48]],
                    start_date=favorite_items[-1].captured_at,
                    end_date=favorite_items[0].captured_at,
                ),
            )
        ]

    def _build_item(self, spec: MediaAssetSpec, existing: MediaItem | None) -> MediaItem:
        prepared = self._prepare_sync_item(spec, existing)
        return self._build_items_batch([prepared])[0]

    def _prepare_batch(
        self,
        entries: list[tuple[str, MediaAssetSpec, MediaItem | None, str]],
        executor: ThreadPoolExecutor,
    ) -> list[PreparedSyncItem]:
        futures: list[Future[PreparedSyncItem]] = [
            executor.submit(self._prepare_sync_item, spec, existing, analysis_mode=analysis_mode)
            for _, spec, existing, analysis_mode in entries
        ]
        return [future.result() for future in futures]

    def _prepare_sync_item(
        self,
        spec: MediaAssetSpec,
        existing: MediaItem | None,
        *,
        analysis_mode: str = "full",
    ) -> PreparedSyncItem:
        still_image: Image.Image | None = None
        video_frames: list[VideoFrameSample] = []
        video_metadata: dict[str, object] = {}

        if spec.media_kind in {"image", "gif", "live_photo"}:
            still_image = self.vision.load_analysis_image(spec)
        if spec.media_kind in {"video", "live_photo"}:
            sampled_frames, sampled_metadata = self.vision.load_video_analysis_frames(spec)
            video_frames = sampled_frames
            video_metadata = sampled_metadata
            if spec.media_kind == "video" and sampled_frames:
                still_image = sampled_frames[0].image.copy()

        if analysis_mode == "human_faces_only" and existing is not None:
            metadata = {
                "captured_at": existing.captured_at,
                "width": existing.width,
                "height": existing.height,
                "duration_seconds": existing.duration_seconds,
                "metadata": {
                    key: value
                    for key, value in existing.metadata.items()
                    if not key.startswith("human_face_")
                },
            }
            thumbnail_path = existing.thumbnail_path
        else:
            metadata = self._extract_metadata(spec)
            thumbnail_source = still_image.copy() if still_image is not None else None
            thumbnail_path = self._ensure_thumbnail_from_image(spec, thumbnail_source)
        return PreparedSyncItem(
            spec=spec,
            existing=existing,
            analysis_mode=analysis_mode,
            metadata=metadata,
            thumbnail_path=thumbnail_path,
            still_image=still_image,
            video_frames=video_frames,
            video_metadata=video_metadata,
        )

    def _build_items_batch(self, prepared_items: list[PreparedSyncItem]) -> list[MediaItem]:
        if not prepared_items:
            return []

        analyses: list[AnalysisResult | None] = [None] * len(prepared_items)
        for analysis_mode in ("full", "full_no_pets", "human_faces_only"):
            batch_indices = [
                index
                for index, prepared in enumerate(prepared_items)
                if prepared.analysis_mode == analysis_mode
            ]
            if not batch_indices:
                continue
            batch_analyses = self.vision.analyze_batch(
                [
                    PreparedAssetInput(
                        spec=prepared_items[index].spec,
                        still_image=(
                            None
                            if prepared_items[index].spec.media_kind == "video"
                            else prepared_items[index].still_image
                        ),
                        video_frames=prepared_items[index].video_frames,
                        video_metadata=prepared_items[index].video_metadata,
                    )
                    for index in batch_indices
                ],
                analysis_mode=analysis_mode,
            )
            for item_index, analysis in zip(batch_indices, batch_analyses, strict=False):
                analyses[item_index] = analysis

        built_items: list[MediaItem] = []
        for prepared, analysis in zip(prepared_items, analyses, strict=False):
            if analysis is None:
                raise RuntimeError("Missing analysis result for prepared media item.")
            if prepared.analysis_mode == "human_faces_only":
                built_items.append(self._build_item_from_face_reanalysis(prepared, analysis))
            else:
                built_items.append(self._build_item_from_prepared_analysis(prepared, analysis))
        return built_items

    def _build_item_from_prepared_analysis(
        self,
        prepared: PreparedSyncItem,
        analysis: AnalysisResult,
    ) -> MediaItem:
        spec = prepared.spec
        existing = prepared.existing
        metadata = prepared.metadata
        tags = sorted(
            {
                spec.media_kind,
                *analysis.tags,
                *self._tags_from_relative_key(spec.relative_key),
            }
        )

        item = MediaItem(
            id=spec.id,
            path=spec.display_path,
            component_paths=spec.component_paths,
            relative_key=spec.relative_key,
            title=spec.title,
            media_kind=spec.media_kind,
            extension=spec.extension,
            file_signature=spec.file_signature,
            size_bytes=spec.size_bytes,
            modified_ts=spec.modified_ts,
            captured_at=metadata["captured_at"],
            discovered_at=existing.discovered_at if existing else utc_now(),
            thumbnail_path=prepared.thumbnail_path,
            width=metadata["width"],
            height=metadata["height"],
            duration_seconds=metadata["duration_seconds"],
            favorite=existing.favorite if existing else False,
            hidden=existing.hidden if existing else False,
            tags=tags,
            detections=analysis.detections,
            manual_persona_ids=[
                persona_id
                for persona_id in (existing.manual_persona_ids if existing else [])
                if persona_id in self.state.personas
            ],
            notes=existing.notes if existing else "",
            metadata=metadata["metadata"] | analysis.metadata,
        )
        self._merge_detection_assignments(item, existing)
        self._auto_assign_personas(item)
        return item

    def _build_item_from_face_reanalysis(
        self,
        prepared: PreparedSyncItem,
        analysis: AnalysisResult,
    ) -> MediaItem:
        existing = prepared.existing
        if existing is None:
            return self._build_item_from_prepared_analysis(prepared, analysis)

        metadata = prepared.metadata
        tags = set(existing.tags)
        tags.discard("face")
        tags.discard("person")
        if analysis.detections:
            tags.update({"face", "person"})

        preserved_detections = [
            self._clone_detection_region(detection)
            for detection in existing.detections
            if detection.kind != "face"
        ]
        face_metadata = {
            key: value
            for key, value in analysis.metadata.items()
            if key.startswith("human_face_") or key in {"analyzed_width", "analyzed_height"}
        }
        preserved_metadata = {
            key: value
            for key, value in existing.metadata.items()
            if not key.startswith("human_face_")
        }
        item = MediaItem(
            id=existing.id,
            path=existing.path,
            component_paths=list(existing.component_paths),
            relative_key=existing.relative_key,
            title=existing.title,
            media_kind=existing.media_kind,
            extension=existing.extension,
            file_signature=existing.file_signature,
            size_bytes=existing.size_bytes,
            modified_ts=existing.modified_ts,
            captured_at=str(metadata.get("captured_at", existing.captured_at)),
            discovered_at=existing.discovered_at,
            thumbnail_path=existing.thumbnail_path,
            width=int(metadata.get("width", existing.width)),
            height=int(metadata.get("height", existing.height)),
            duration_seconds=float(metadata.get("duration_seconds", existing.duration_seconds)),
            favorite=existing.favorite,
            hidden=existing.hidden,
            tags=sorted(tags),
            detections=[*analysis.detections, *preserved_detections],
            manual_persona_ids=[
                persona_id
                for persona_id in existing.manual_persona_ids
                if persona_id in self.state.personas
            ],
            notes=existing.notes,
            metadata=preserved_metadata | face_metadata,
        )
        self._merge_detection_assignments(item, existing)
        self._auto_assign_personas(item)
        return item

    def _extract_metadata(self, spec: MediaAssetSpec) -> dict[str, object]:
        captured_at = self._guess_capture_date(spec)
        width = 0
        height = 0
        duration_seconds = 0.0
        metadata: dict[str, object] = {"component_count": len(spec.component_paths)}

        image_path = self.vision.primary_image_path(spec)
        if image_path:
            try:
                with Image.open(image_path) as image:
                    image = ImageOps.exif_transpose(image)
                    width, height = image.size
                    exif = image.getexif()
                    captured_at = self._parse_exif_datetime(
                        exif.get(36867) or exif.get(306)
                    ) or captured_at
                    metadata["animated"] = bool(getattr(image, "is_animated", False))
                    if spec.media_kind == "gif":
                        try:
                            duration_seconds = sum(
                                (frame.info.get("duration", 0) for frame in ImageSequence.Iterator(image))
                            ) / 1000.0
                        except Exception:
                            duration_seconds = 0.0
            except (UnidentifiedImageError, OSError):
                pass

        video_path = self.vision.primary_video_path(spec)
        if video_path and cv2 is not None:
            capture = cv2.VideoCapture(str(video_path))
            try:
                width = width or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = height or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                if fps > 0 and frame_count > 0:
                    duration_seconds = max(duration_seconds, frame_count / fps)
            finally:
                capture.release()

        return {
            "captured_at": captured_at,
            "width": width,
            "height": height,
            "duration_seconds": duration_seconds,
            "metadata": metadata,
        }

    def _ensure_thumbnail(self, spec: MediaAssetSpec) -> str:
        cache_path = self.config.cache_path / f"{spec.id}.jpg"
        image = self.vision.load_preview_image(spec)
        if image is None:
            return ""

        return self._ensure_thumbnail_from_image(spec, image)

    def _ensure_thumbnail_from_image(self, spec: MediaAssetSpec, image: Image.Image | None) -> str:
        if image is None:
            return ""
        cache_path = self.config.cache_path / f"{spec.id}.jpg"
        image.thumbnail((self.config.thumbnail_size, self.config.thumbnail_size))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(cache_path, format="JPEG", quality=88)
        return str(cache_path)

    def _clone_detection_region(self, detection: DetectionRegion) -> DetectionRegion:
        return DetectionRegion(
            id=detection.id,
            kind=detection.kind,
            label=detection.label,
            confidence=detection.confidence,
            bbox=list(detection.bbox),
            persona_id=detection.persona_id,
            encoding=list(detection.encoding),
            signature=detection.signature,
        )

    def _merge_detection_assignments(
        self,
        item: MediaItem,
        existing: MediaItem | None,
    ) -> None:
        if not existing:
            return
        existing_regions = {region.id: region for region in existing.detections}
        matched_region_ids: set[str] = set()
        for detection in item.detections:
            prior = existing_regions.get(detection.id)
            if prior and prior.persona_id in self.state.personas:
                detection.persona_id = prior.persona_id
                matched_region_ids.add(prior.id)
                continue
            prior = self._best_previous_detection_match(
                detection,
                existing.detections,
                matched_region_ids,
            )
            if prior and prior.persona_id in self.state.personas:
                detection.persona_id = prior.persona_id
                matched_region_ids.add(prior.id)

    def _auto_assign_personas(self, item: MediaItem) -> None:
        for detection in item.detections:
            if detection.persona_id:
                continue
            if detection.kind == "face" and detection.encoding:
                persona_id = self._match_face_to_persona(detection.encoding)
                if persona_id:
                    detection.persona_id = persona_id
            elif self._is_pet_detection(detection):
                persona_id = self._match_pet_to_persona(detection.encoding, detection.signature)
                if persona_id:
                    detection.persona_id = persona_id

    def _match_face_to_persona(self, encoding: list[float]) -> str | None:
        best_persona_id = None
        best_similarity = -1.0
        for persona in self.list_personas(kind="person"):
            for candidate in persona.reference_encodings:
                similarity = self._cosine_similarity(encoding, candidate)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_persona_id = persona.id
        if best_similarity >= self.config.face_embedding_similarity_threshold:
            return best_persona_id
        best_distance = math.inf
        for persona in self.list_personas(kind="person"):
            for candidate in persona.reference_encodings:
                distance = self._euclidean_distance(encoding, candidate)
                if distance < best_distance:
                    best_distance = distance
                    best_persona_id = persona.id
        if best_distance <= self.config.face_match_threshold:
            return best_persona_id
        return None

    def _reanalysis_mode(self, item: MediaItem | None) -> str:
        if item is None:
            return "full"
        if self.config.face_recognition_enabled:
            recorded_revision = str(item.metadata.get("human_face_pipeline", ""))
            current_revision = self._current_human_face_pipeline_revision()
            if current_revision and recorded_revision != current_revision:
                return "human_faces_only"
        return "none"

    def _current_human_face_pipeline_revision(self) -> str:
        if not self.config.face_recognition_enabled:
            return ""
        vision = self.vision
        if vision.human_face_backend is None:
            return ""
        return vision.human_face_pipeline_id

    def _best_previous_detection_match(
        self,
        detection: DetectionRegion,
        prior_detections: list[DetectionRegion],
        matched_region_ids: set[str],
    ) -> DetectionRegion | None:
        best_candidate: DetectionRegion | None = None
        best_score = -1.0
        for prior in prior_detections:
            if prior.id in matched_region_ids:
                continue
            if prior.kind != detection.kind:
                continue

            score = -1.0
            if detection.kind == "face":
                if detection.encoding and prior.encoding:
                    similarity = self._cosine_similarity(detection.encoding, prior.encoding)
                    if similarity >= 0.50:
                        score = 2.0 + similarity
                elif detection.signature and prior.signature:
                    distance = self._signature_distance(detection.signature, prior.signature)
                    if distance <= 10:
                        score = 1.5 - (distance / 20.0)
            elif self._is_pet_detection(detection) and self._is_pet_detection(prior):
                if detection.encoding and prior.encoding:
                    similarity = self._cosine_similarity(detection.encoding, prior.encoding)
                    if similarity >= 0.60:
                        score = 2.0 + similarity
                elif detection.signature and prior.signature:
                    distance = self._signature_distance(detection.signature, prior.signature)
                    if distance <= 10:
                        score = 1.5 - (distance / 20.0)

            iou_score = self._iou(detection.bbox, prior.bbox)
            if iou_score >= 0.55:
                score = max(score, 1.0 + iou_score)

            if score > best_score:
                best_score = score
                best_candidate = prior
        return best_candidate

    def _iou(self, left_bbox: list[int], right_bbox: list[int]) -> float:
        left_x1, left_y1, left_width, left_height = left_bbox
        right_x1, right_y1, right_width, right_height = right_bbox
        left_x2 = left_x1 + left_width
        left_y2 = left_y1 + left_height
        right_x2 = right_x1 + right_width
        right_y2 = right_y1 + right_height

        inter_x1 = max(left_x1, right_x1)
        inter_y1 = max(left_y1, right_y1)
        inter_x2 = min(left_x2, right_x2)
        inter_y2 = min(left_y2, right_y2)
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        if inter_width == 0 or inter_height == 0:
            return 0.0

        intersection = inter_width * inter_height
        left_area = max(1, left_width * left_height)
        right_area = max(1, right_width * right_height)
        union = left_area + right_area - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _match_pet_to_persona(self, embedding: list[float], signature: str | None) -> str | None:
        if embedding:
            best_persona_id = None
            best_similarity = -1.0
            for persona in self.list_personas(kind="pet"):
                for candidate in persona.reference_encodings:
                    similarity = self._cosine_similarity(embedding, candidate)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_persona_id = persona.id
            if best_similarity >= self.config.pet_embedding_similarity_threshold:
                return best_persona_id

        if not signature:
            return None

        best_persona_id = None
        best_distance = math.inf
        for persona in self.list_personas(kind="pet"):
            for candidate in persona.reference_signatures:
                distance = self._signature_distance(signature, candidate)
                if distance < best_distance:
                    best_distance = distance
                    best_persona_id = persona.id
        if best_distance <= self.config.pet_hash_distance_threshold:
            return best_persona_id
        return None

    def _build_unknown_clusters(self, kind: str) -> list[UnknownPersonaCluster]:
        records = self.store.query_detections(cluster_kind=kind, dirty_only=False)
        cluster_states = self._build_unknown_cluster_states_from_detection_records(
            kind,
            records,
            merge_states=True,
        )
        return [
            self._finalize_unknown_cluster(
                cluster_state,
                revision=self.state.updated_at,
                is_partial=False,
            )
            for cluster_state in cluster_states
            if cluster_state.members
        ]

    def _refresh_unknown_clusters_kind(
        self,
        kind: str,
        *,
        partial: bool,
        progress_callback: Callable[[ProgressUpdate], None] | None,
        overall_started_at: float,
        overall_current: int,
        overall_total: int,
    ) -> list[UnknownPersonaCluster]:
        revision = self.state.updated_at
        kind_started_at = monotonic()
        self._emit_progress(
            progress_callback,
            self._make_progress_update(
                phase="unknown_clusters",
                message=f"Rebuilding {'partial' if partial else 'final'} {kind} clusters",
                current=overall_current,
                total=overall_total,
                detail=f"revision {revision}",
                indeterminate=True,
                overall_started_at=overall_started_at,
                step_started_at=kind_started_at,
            ),
        )
        dirty_count = self.store.count_detections(kind, dirty_only=True)
        existing_cluster_payloads = self.store.list_unknown_clusters(kind)
        if partial:
            built_clusters = self._rebuild_unknown_clusters_incremental(
                kind,
                revision=revision,
                progress_callback=progress_callback,
                overall_started_at=overall_started_at,
                step_started_at=kind_started_at,
            )
        elif dirty_count > 0 or existing_cluster_payloads:
            built_clusters = self._rebuild_unknown_clusters_incremental(
                kind,
                revision=revision,
                progress_callback=progress_callback,
                overall_started_at=overall_started_at,
                step_started_at=kind_started_at,
            )
            self.store.mark_unknown_clusters_stable(kind, revision=revision)
            built_clusters = [
                self._deserialize_unknown_cluster(entry)
                for entry in self.store.list_unknown_clusters(kind)
            ]
        else:
            built_clusters = self._rebuild_unknown_clusters_full(
                kind,
                revision=revision,
                progress_callback=progress_callback,
                overall_started_at=overall_started_at,
                step_started_at=kind_started_at,
            )
        self._emit_progress(
            progress_callback,
            self._make_progress_update(
                phase="unknown_clusters",
                message=f"Updated {'partial' if partial else 'final'} {kind} clusters",
                current=overall_current + 1,
                total=overall_total,
                detail=f"{len(built_clusters)} cluster(s)",
                overall_started_at=overall_started_at,
                step_started_at=kind_started_at,
            ),
        )
        return built_clusters

    def _rebuild_unknown_clusters_full(
        self,
        kind: str,
        *,
        revision: str,
        progress_callback: Callable[[ProgressUpdate], None] | None,
        overall_started_at: float,
        step_started_at: float,
    ) -> list[UnknownPersonaCluster]:
        records = self.store.query_detections(cluster_kind=kind, dirty_only=False)
        cluster_states = self._build_unknown_cluster_states_from_detection_records(
            kind,
            records,
            merge_states=True,
            progress_callback=progress_callback,
            progress_message=f"Clustering all {kind} detections",
            overall_started_at=overall_started_at,
            step_started_at=step_started_at,
        )
        previous_clusters = self._previous_unknown_clusters_by_id(kind)
        clusters = [
            self._finalize_unknown_cluster(
                cluster_state,
                revision=revision,
                is_partial=False,
                previous_cluster=previous_clusters.get(cluster_state.cluster_id),
            )
            for cluster_state in cluster_states
            if cluster_state.members
        ]
        self._persist_unknown_clusters(
            kind,
            clusters,
            revision=revision,
            partial=False,
            previous_clusters=previous_clusters,
        )
        self.store.mark_detections_cluster_clean(
            [(record.item_id, record.detection_id) for record in records],
            cleaned_revision=revision,
        )
        return clusters

    def _rebuild_unknown_clusters_incremental(
        self,
        kind: str,
        *,
        revision: str,
        progress_callback: Callable[[ProgressUpdate], None] | None,
        overall_started_at: float,
        step_started_at: float,
    ) -> list[UnknownPersonaCluster]:
        dirty_records = self.store.query_detections(cluster_kind=kind, dirty_only=True)
        if not dirty_records:
            return [
                self._deserialize_unknown_cluster(entry)
                for entry in self.store.list_unknown_clusters(kind)
            ]
        states_by_id = {
            cluster_state.cluster_id: cluster_state
            for cluster_state in self._load_persisted_unknown_cluster_states(kind)
        }
        detection_to_cluster: dict[tuple[str, str], str] = {}
        for cluster_state in states_by_id.values():
            for member in cluster_state.members:
                detection_to_cluster[(member.item_id, member.region_id)] = cluster_state.cluster_id

        total_records = len(dirty_records)
        processed_records: list[DetectionRecord] = []
        latest_clusters: list[UnknownPersonaCluster] = []
        for index, record in enumerate(dirty_records, start=1):
            detection_key = (record.item_id, record.detection_id)
            existing_cluster_id = detection_to_cluster.pop(detection_key, "")
            if existing_cluster_id:
                cluster_state = states_by_id.get(existing_cluster_id)
                if cluster_state is not None:
                    self._remove_unknown_cluster_member(cluster_state, detection_key)

            if self._is_unknown_cluster_candidate_record(kind, record):
                matched_cluster = self._best_unknown_cluster_match(
                    states_by_id.values(),
                    kind,
                    record.to_region(),
                )
                if matched_cluster is None:
                    matched_cluster = _UnknownClusterState(
                        cluster_id=stable_id(f"unknown-cluster:{kind}:{record.item_id}:{record.detection_id}"),
                        kind=kind,
                    )
                    states_by_id[matched_cluster.cluster_id] = matched_cluster
                self._append_unknown_cluster_member_from_record(matched_cluster, record)
                detection_to_cluster[detection_key] = matched_cluster.cluster_id

            processed_records.append(record)
            if (
                progress_callback is not None
                and total_records > 0
                and (index == total_records or index % 128 == 0)
            ):
                self._emit_progress(
                    progress_callback,
                    self._make_progress_update(
                        phase="unknown_clusters",
                        message=f"Applying live {kind} cluster updates",
                        current=index,
                        total=total_records,
                        detail=f"{len(states_by_id)} active cluster(s)",
                        overall_started_at=overall_started_at,
                        step_started_at=step_started_at,
                    ),
                )
            if index % CLUSTER_DIRTY_CHUNK_SIZE == 0:
                latest_clusters = self._persist_incremental_unknown_cluster_checkpoint(
                    kind,
                    states_by_id,
                    processed_records,
                    revision=revision,
                    partial=True,
                )
                processed_records = []

        latest_clusters = self._persist_incremental_unknown_cluster_checkpoint(
            kind,
            states_by_id,
            processed_records,
            revision=revision,
            partial=True,
        )
        return latest_clusters

    def _persist_incremental_unknown_cluster_checkpoint(
        self,
        kind: str,
        states_by_id: dict[str, _UnknownClusterState],
        processed_records: list[DetectionRecord],
        *,
        revision: str,
        partial: bool,
    ) -> list[UnknownPersonaCluster]:
        cluster_states = [state for state in states_by_id.values() if state.members]
        previous_clusters = self._previous_unknown_clusters_by_id(kind)
        clusters = [
            self._finalize_unknown_cluster(
                cluster_state,
                revision=revision,
                is_partial=partial,
                previous_cluster=previous_clusters.get(cluster_state.cluster_id),
            )
            for cluster_state in cluster_states
        ]
        self._persist_unknown_clusters(
            kind,
            clusters,
            revision=revision,
            partial=partial,
            previous_clusters=previous_clusters,
        )
        if processed_records:
            self.store.mark_detections_cluster_clean(
                [(record.item_id, record.detection_id) for record in processed_records],
                cleaned_revision=revision,
            )
        return clusters

    def _build_unknown_cluster_states_from_detection_records(
        self,
        kind: str,
        records: list[DetectionRecord],
        *,
        merge_states: bool,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
        progress_message: str = "",
        overall_started_at: float | None = None,
        step_started_at: float | None = None,
    ) -> list[_UnknownClusterState]:
        cluster_states: list[_UnknownClusterState] = []
        total_records = len(records)
        for index, record in enumerate(records, start=1):
            if not self._is_unknown_cluster_candidate_record(kind, record):
                continue
            self._add_detection_record_to_unknown_clusters(cluster_states, kind, record)
            if (
                progress_callback is not None
                and total_records > 0
                and (index == total_records or index % 128 == 0)
            ):
                self._emit_progress(
                    progress_callback,
                    self._make_progress_update(
                        phase="unknown_clusters",
                        message=progress_message or f"Clustering {kind} detections",
                        current=index,
                        total=total_records,
                        detail=f"{len(cluster_states)} provisional cluster(s)",
                        overall_started_at=overall_started_at,
                        step_started_at=step_started_at,
                    ),
                )
        if merge_states:
            cluster_states = self._merge_unknown_cluster_states(cluster_states, kind)
        return [cluster_state for cluster_state in cluster_states if cluster_state.members]

    def _load_persisted_unknown_cluster_states(self, kind: str) -> list[_UnknownClusterState]:
        cluster_states: list[_UnknownClusterState] = []
        for cluster_payload in self.store.load_unknown_cluster_states(kind):
            cluster_state = _UnknownClusterState(
                cluster_id=str(cluster_payload["id"]),
                kind=str(cluster_payload["kind"]),
            )
            for record in cluster_payload.get("detections", []):
                if isinstance(record, DetectionRecord):
                    self._append_unknown_cluster_member_from_record(cluster_state, record)
            if cluster_state.members:
                cluster_states.append(cluster_state)
        return cluster_states

    def _unknown_cluster_state_by_id(self, cluster_id: str) -> _UnknownClusterState | None:
        if not cluster_id:
            return None
        for kind in ("person", "pet"):
            for cluster_state in self._load_persisted_unknown_cluster_states(kind):
                if cluster_state.cluster_id == cluster_id:
                    return cluster_state
        return None

    def _score_unknown_cluster_for_persona(
        self,
        cluster: _UnknownClusterState,
        persona: Persona,
    ) -> tuple[float, str]:
        best_embedding_similarity = -1.0
        if cluster.embeddings and persona.reference_encodings:
            best_embedding_similarity = max(
                self._cosine_similarity(cluster_embedding, reference_embedding)
                for cluster_embedding in cluster.embeddings
                for reference_embedding in persona.reference_encodings
            )
            threshold = (
                self.config.face_embedding_similarity_threshold
                if cluster.kind == "person"
                else self.config.pet_embedding_similarity_threshold
            )
            if best_embedding_similarity >= threshold:
                return best_embedding_similarity, "embedding"

        if cluster.kind != "pet" or not cluster.signatures or not persona.reference_signatures:
            return 0.0, ""

        best_signature_distance = min(
            self._signature_distance(cluster_signature, reference_signature)
            for cluster_signature in cluster.signatures
            for reference_signature in persona.reference_signatures
        )
        if best_signature_distance > self.config.pet_hash_distance_threshold:
            return 0.0, ""
        score = 1.0 - (best_signature_distance / max(1, self.config.pet_hash_distance_threshold))
        return max(score, 0.0), "signature"

    def _is_unknown_cluster_candidate(self, kind: str, detection: DetectionRegion) -> bool:
        if kind == "person":
            return detection.kind == "face" and bool(detection.encoding or detection.signature)
        if kind == "pet":
            return self._is_pet_detection(detection) and bool(detection.encoding or detection.signature)
        return False

    def _is_unknown_cluster_candidate_record(self, kind: str, record: DetectionRecord) -> bool:
        if record.persona_id:
            return False
        return self._is_unknown_cluster_candidate(kind, record.to_region())

    def _add_detection_record_to_unknown_clusters(
        self,
        clusters: list[_UnknownClusterState],
        kind: str,
        record: DetectionRecord,
    ) -> None:
        matched_cluster = self._best_unknown_cluster_match(clusters, kind, record.to_region())
        if matched_cluster is None:
            matched_cluster = _UnknownClusterState(
                cluster_id=stable_id(f"unknown-cluster:{kind}:{record.item_id}:{record.detection_id}"),
                kind=kind,
            )
            clusters.append(matched_cluster)
        self._append_unknown_cluster_member_from_record(matched_cluster, record)

    def _best_unknown_cluster_match(
        self,
        clusters: Iterable[_UnknownClusterState],
        kind: str,
        detection: DetectionRegion,
    ) -> _UnknownClusterState | None:
        embedding_match: _UnknownClusterState | None = None
        best_similarity = -1.0
        signature_match: _UnknownClusterState | None = None
        best_distance = math.inf
        embedding_threshold = self._unknown_cluster_similarity_threshold(kind)
        signature_threshold = self._unknown_cluster_signature_threshold(kind)

        for cluster in clusters:
            if detection.encoding and cluster.embeddings:
                similarity = max(
                    self._cosine_similarity(detection.encoding, candidate)
                    for candidate in cluster.embeddings
                )
                if similarity >= embedding_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    embedding_match = cluster
            if detection.signature and cluster.signatures:
                distance = min(
                    self._signature_distance(detection.signature, candidate)
                    for candidate in cluster.signatures
                )
                if distance <= signature_threshold and distance < best_distance:
                    best_distance = distance
                    signature_match = cluster

        return embedding_match or signature_match

    def _append_unknown_cluster_member_from_record(
        self,
        cluster: _UnknownClusterState,
        record: DetectionRecord,
    ) -> None:
        detection_key = (record.item_id, record.detection_id)
        if any(
            existing.item_id == detection_key[0] and existing.region_id == detection_key[1]
            for existing in cluster.members
        ):
            return
        detection = record.to_region()
        cluster.members.append(
            UnknownClusterMember(
                item_id=record.item_id,
                region_id=record.detection_id,
                label=record.label,
                confidence=record.confidence,
                captured_at=record.captured_at,
            )
        )
        cluster.detection_records.append(record)
        cluster.item_ids.add(record.item_id)
        cluster.labels[record.label.lower()] += 1
        if record.encoding:
            self._remember_cluster_embedding(cluster, record.encoding)
        if record.signature:
            self._remember_cluster_signature(cluster, record.signature)
        if (
            record.confidence > cluster.representative_confidence
            or not cluster.representative_item_id
        ):
            cluster.representative_item_id = record.item_id
            cluster.representative_region_id = record.detection_id
            cluster.representative_confidence = record.confidence
        if record.captured_at > cluster.latest_captured_at:
            cluster.latest_captured_at = record.captured_at

    def _remove_unknown_cluster_member(
        self,
        cluster: _UnknownClusterState,
        detection_key: tuple[str, str],
    ) -> None:
        remaining_records = [
            record
            for record in cluster.detection_records
            if (record.item_id, record.detection_id) != detection_key
        ]
        cluster.members = []
        cluster.detection_records = []
        cluster.item_ids = set()
        cluster.labels = Counter()
        cluster.embeddings = []
        cluster.signatures = []
        cluster.representative_item_id = ""
        cluster.representative_region_id = ""
        cluster.representative_confidence = -1.0
        cluster.latest_captured_at = ""
        for record in remaining_records:
            self._append_unknown_cluster_member_from_record(cluster, record)

    def _merge_unknown_cluster_states(
        self,
        clusters: list[_UnknownClusterState],
        kind: str,
    ) -> list[_UnknownClusterState]:
        if len(clusters) < 2:
            return clusters

        merged = True
        while merged:
            merged = False
            for left_index, left in enumerate(clusters):
                for right_index in range(left_index + 1, len(clusters)):
                    right = clusters[right_index]
                    if not self._should_merge_unknown_clusters(kind, left, right):
                        continue
                    self._merge_unknown_cluster_state(left, right)
                    clusters.pop(right_index)
                    merged = True
                    break
                if merged:
                    break
        return clusters

    def _should_merge_unknown_clusters(
        self,
        kind: str,
        left: _UnknownClusterState,
        right: _UnknownClusterState,
    ) -> bool:
        if kind == "pet" and not self._pet_cluster_labels_compatible(left, right):
            return False

        similarity_threshold = self._unknown_cluster_merge_similarity_threshold(kind)
        signature_threshold = self._unknown_cluster_merge_signature_threshold(kind)

        if left.embeddings and right.embeddings:
            best_similarity = max(
                self._cosine_similarity(left_encoding, right_encoding)
                for left_encoding in left.embeddings
                for right_encoding in right.embeddings
            )
            if best_similarity >= similarity_threshold:
                return True

        if left.signatures and right.signatures:
            best_distance = min(
                self._signature_distance(left_signature, right_signature)
                for left_signature in left.signatures
                for right_signature in right.signatures
            )
            if best_distance <= signature_threshold:
                return True

        return False

    def _merge_unknown_cluster_state(
        self,
        target: _UnknownClusterState,
        source: _UnknownClusterState,
    ) -> None:
        target.members.extend(source.members)
        target.detection_records.extend(source.detection_records)
        target.item_ids.update(source.item_ids)
        target.labels.update(source.labels)
        for encoding in source.embeddings:
            self._remember_cluster_embedding(target, encoding)
        for signature in source.signatures:
            self._remember_cluster_signature(target, signature)
        if source.representative_confidence > target.representative_confidence:
            target.representative_item_id = source.representative_item_id
            target.representative_region_id = source.representative_region_id
            target.representative_confidence = source.representative_confidence
        if source.latest_captured_at > target.latest_captured_at:
            target.latest_captured_at = source.latest_captured_at

    def _pet_cluster_labels_compatible(
        self,
        left: _UnknownClusterState,
        right: _UnknownClusterState,
    ) -> bool:
        left_labels = set(left.labels)
        right_labels = set(right.labels)
        if "pet" in left_labels or "pet" in right_labels:
            return True
        if left_labels & CAT_LABELS and right_labels & CAT_LABELS:
            return True
        if left_labels & DOG_LABELS and right_labels & DOG_LABELS:
            return True
        return bool(left_labels & right_labels)

    def _remember_cluster_embedding(
        self,
        cluster: _UnknownClusterState,
        encoding: list[float],
    ) -> None:
        if not cluster.embeddings:
            cluster.embeddings.append(encoding)
            return
        closest = max(
            self._cosine_similarity(encoding, candidate)
            for candidate in cluster.embeddings
        )
        if closest < REFERENCE_EMBEDDING_SIMILARITY_THRESHOLD:
            cluster.embeddings.append(encoding)

    def _remember_cluster_signature(
        self,
        cluster: _UnknownClusterState,
        signature: str,
    ) -> None:
        if not cluster.signatures:
            cluster.signatures.append(signature)
            return
        closest = min(
            self._signature_distance(signature, candidate)
            for candidate in cluster.signatures
        )
        if closest > REFERENCE_SIGNATURE_DISTANCE_THRESHOLD:
            cluster.signatures.append(signature)

    def _finalize_unknown_cluster(
        self,
        cluster: _UnknownClusterState,
        *,
        revision: str,
        is_partial: bool,
        previous_cluster: dict[str, object] | None = None,
    ) -> UnknownPersonaCluster:
        member_ids = sorted(
            {(member.item_id, member.region_id) for member in cluster.members},
            key=lambda entry: entry[0] + entry[1],
        )
        representative_item = self.state.items.get(cluster.representative_item_id)
        representative_detection = None
        if representative_item is not None:
            representative_detection = next(
                (entry for entry in representative_item.detections if entry.id == cluster.representative_region_id),
                None,
            )

        preview_path = ""
        if representative_item is not None and representative_detection is not None:
            preview_path = self._cluster_preview_path_for_representative(
                cluster.cluster_id,
                representative_item,
                representative_detection,
                previous_cluster=previous_cluster,
            )

        average_confidence = sum(member.confidence for member in cluster.members) / max(1, len(cluster.members))
        label = cluster.labels.most_common(1)[0][0] if cluster.labels else ("face" if cluster.kind == "person" else "pet")
        return UnknownPersonaCluster(
            id=cluster.cluster_id,
            kind=cluster.kind,
            label=label,
            member_count=len(member_ids),
            item_count=len(cluster.item_ids),
            member_ids=member_ids,
            item_ids=sorted(cluster.item_ids),
            preview_path=preview_path,
            latest_captured_at=cluster.latest_captured_at,
            average_confidence=average_confidence,
            representative_item_id=cluster.representative_item_id,
            representative_detection_id=cluster.representative_region_id,
            revision=revision,
            is_partial=is_partial,
            updated_at=revision,
        )

    def _persist_unknown_clusters(
        self,
        kind: str,
        clusters: list[UnknownPersonaCluster],
        *,
        revision: str,
        partial: bool,
        previous_clusters: dict[str, dict[str, object]] | None = None,
    ) -> None:
        previous_clusters = previous_clusters or self._previous_unknown_clusters_by_id(kind)
        serialized_clusters = [self._serialize_unknown_cluster(cluster) for cluster in clusters]
        self.store.replace_unknown_clusters(
            kind,
            serialized_clusters,
            revision=revision,
            partial=partial,
        )
        retained_cluster_ids = {cluster.id for cluster in clusters}
        removed_preview_paths = [
            str(cluster.get("preview_path", ""))
            for cluster_id, cluster in previous_clusters.items()
            if cluster_id not in retained_cluster_ids
        ]
        self._cleanup_orphaned_cluster_previews(removed_preview_paths)

    def _previous_unknown_clusters_by_id(self, kind: str) -> dict[str, dict[str, object]]:
        return {
            str(cluster["id"]): cluster
            for cluster in self.store.list_unknown_clusters(kind)
        }

    def _cleanup_orphaned_cluster_previews(self, preview_paths: Iterable[str]) -> None:
        for preview_path in preview_paths:
            if not preview_path:
                continue
            try:
                Path(preview_path).unlink(missing_ok=True)
            except OSError:
                continue

    def _unknown_cluster_similarity_threshold(self, kind: str) -> float:
        if kind == "pet":
            return min(
                UNKNOWN_PET_CLUSTER_SIMILARITY_CAP,
                max(
                    UNKNOWN_PET_CLUSTER_SIMILARITY_FLOOR,
                    self.config.pet_embedding_similarity_threshold + UNKNOWN_PET_CLUSTER_SIMILARITY_BOOST,
                ),
            )
        return min(
            UNKNOWN_PERSON_CLUSTER_SIMILARITY_CAP,
            max(
                UNKNOWN_PERSON_CLUSTER_SIMILARITY_FLOOR,
                self.config.face_embedding_similarity_threshold + UNKNOWN_PERSON_CLUSTER_SIMILARITY_BOOST,
            ),
        )

    def _unknown_cluster_merge_similarity_threshold(self, kind: str) -> float:
        if kind == "pet":
            return max(
                UNKNOWN_PET_CLUSTER_SIMILARITY_FLOOR - 0.02,
                self._unknown_cluster_similarity_threshold(kind) - UNKNOWN_PET_CLUSTER_MERGE_MARGIN,
            )
        return max(
            UNKNOWN_PERSON_CLUSTER_SIMILARITY_FLOOR - 0.04,
            self._unknown_cluster_similarity_threshold(kind) - UNKNOWN_PERSON_CLUSTER_MERGE_MARGIN,
        )

    def _unknown_cluster_signature_threshold(self, kind: str) -> float:
        if kind == "pet":
            return min(float(self.config.pet_hash_distance_threshold), REFERENCE_SIGNATURE_DISTANCE_THRESHOLD)
        return REFERENCE_SIGNATURE_DISTANCE_THRESHOLD

    def _unknown_cluster_merge_signature_threshold(self, kind: str) -> float:
        base = self._unknown_cluster_signature_threshold(kind)
        return base + UNKNOWN_CLUSTER_SIGNATURE_MERGE_SLACK

    def _cluster_preview_path_for_representative(
        self,
        cluster_id: str,
        item: MediaItem,
        detection: DetectionRegion,
        *,
        previous_cluster: dict[str, object] | None,
    ) -> str:
        if previous_cluster:
            previous_preview = str(previous_cluster.get("preview_path", ""))
            same_representative = (
                str(previous_cluster.get("representative_item_id", "")) == item.id
                and str(previous_cluster.get("representative_detection_id", "")) == detection.id
            )
            if same_representative and previous_preview and Path(previous_preview).exists():
                return previous_preview
        return self._ensure_cluster_preview(cluster_id, item, detection)

    def _ensure_cluster_preview(
        self,
        cluster_id: str,
        item: MediaItem,
        detection: DetectionRegion,
    ) -> str:
        preview_root = self._cluster_previews_root()
        preview_root.mkdir(parents=True, exist_ok=True)
        preview_path = preview_root / f"{cluster_id}.jpg"
        crop = self._extract_reference_crop(item, detection)
        if crop is None:
            return item.thumbnail_path
        image = crop.convert("RGB")
        image.thumbnail((256, 256))
        image.save(preview_path, format="JPEG", quality=90)
        return str(preview_path)

    def _cleanup_collections(self) -> bool:
        changed = False
        live_item_ids = set(self.state.items)
        for album in self.state.albums.values():
            item_ids = [item_id for item_id in album.item_ids if item_id in live_item_ids]
            if item_ids != album.item_ids:
                album.item_ids = item_ids
                changed = True
        for memory in self.state.memories.values():
            item_ids = [item_id for item_id in memory.item_ids if item_id in live_item_ids]
            if item_ids != memory.item_ids:
                memory.item_ids = item_ids
                changed = True
        for persona in self.state.personas.values():
            if persona.avatar_item_id and persona.avatar_item_id not in live_item_ids:
                persona.avatar_item_id = ""
                changed = True
            existing_references = [
                reference
                for reference in persona.reference_images
                if reference.get("path") and Path(reference["path"]).exists()
            ]
            if existing_references != persona.reference_images:
                changed = True
            deduped_references = self._dedupe_reference_images(persona, existing_references)
            if deduped_references != existing_references:
                changed = True
            persona.reference_images = deduped_references
        return changed

    def _resolve_persona(self, persona_id: str, new_name: str, kind: str) -> Persona:
        if persona_id:
            persona = self.state.personas.get(persona_id)
            if not persona:
                raise ValueError("Persona not found.")
            return persona
        return self.create_persona(new_name, kind)

    def _remember_embedding(self, persona: Persona, encoding: list[float]) -> bool:
        if not persona.reference_encodings:
            persona.reference_encodings.append(encoding)
            return True
        closest = max(
            self._cosine_similarity(encoding, candidate)
            for candidate in persona.reference_encodings
        )
        if closest < REFERENCE_EMBEDDING_SIMILARITY_THRESHOLD:
            persona.reference_encodings.append(encoding)
            return True
        return False

    def _remember_signature(self, persona: Persona, signature: str) -> bool:
        if signature not in persona.reference_signatures:
            persona.reference_signatures.append(signature)
            return True
        return False

    def _remember_detection_reference(
        self,
        persona: Persona,
        item: MediaItem,
        detection: DetectionRegion,
    ) -> None:
        if not self._can_learn_from_detection(persona, detection):
            return
        if detection.encoding:
            self._remember_embedding(persona, detection.encoding)
        if detection.signature:
            self._remember_signature(persona, detection.signature)
        if not self._should_store_reference_image(persona, detection):
            return
        reference_image = self._extract_reference_crop(item, detection)
        if reference_image is None:
            return
        self._remember_reference_image(persona, item, detection, reference_image)

    def _can_learn_from_detection(self, persona: Persona, detection: DetectionRegion) -> bool:
        if persona.kind == "person":
            return detection.kind == "face"
        if persona.kind == "pet":
            return self._is_pet_detection(detection)
        return False

    def _should_store_reference_image(
        self,
        persona: Persona,
        detection: DetectionRegion,
    ) -> bool:
        if not persona.reference_images:
            return True
        reference_detections = self._reference_source_detections(persona.reference_images)
        if not reference_detections:
            return True
        return self._is_distinct_reference_detection(detection, reference_detections)

    def _remember_reference_image(
        self,
        persona: Persona,
        item: MediaItem,
        detection: DetectionRegion,
        crop: Image.Image,
    ) -> None:
        references_root = self._reference_images_root() / persona.id
        references_root.mkdir(parents=True, exist_ok=True)
        filename = f"{stable_id(f'{persona.id}:{item.id}:{detection.id}:{detection.signature or detection.label}')}.jpg"
        reference_path = references_root / filename

        if not reference_path.exists():
            image = crop.convert("RGB")
            image.thumbnail((384, 384))
            image.save(reference_path, format="JPEG", quality=90)

        reference_entry = {
            "path": str(reference_path),
            "source_item_id": item.id,
            "source_region_id": detection.id,
            "label": detection.label,
            "kind": detection.kind,
            "created_at": utc_now(),
        }
        existing_index = next(
            (
                index
                for index, entry in enumerate(persona.reference_images)
                if entry.get("path") == str(reference_path)
            ),
            None,
        )
        if existing_index is None:
            persona.reference_images.append(reference_entry)
        else:
            persona.reference_images[existing_index] = reference_entry

    def _dedupe_reference_images(
        self,
        persona: Persona,
        references: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        if len(references) <= 1:
            return list(references)

        kept_references: list[dict[str, str]] = []
        kept_detections: list[DetectionRegion] = []
        seen_paths: set[str] = set()
        for reference in sorted(references, key=self._reference_image_sort_key):
            path = reference.get("path", "")
            if not path or path in seen_paths:
                continue
            detection = self._reference_source_detection(reference)
            if detection and not self._is_distinct_reference_detection(detection, kept_detections):
                self._remove_reference_image_file(persona, Path(path), protected_paths=seen_paths)
                continue
            kept_references.append(reference)
            seen_paths.add(path)
            if detection is not None:
                kept_detections.append(detection)
        return kept_references

    def _reference_image_sort_key(self, reference: dict[str, str]) -> tuple[str, str]:
        return (reference.get("created_at", ""), reference.get("path", ""))

    def _reference_source_detection(self, reference: dict[str, str]) -> DetectionRegion | None:
        item_id = reference.get("source_item_id", "")
        region_id = reference.get("source_region_id", "")
        if not item_id or not region_id:
            return None
        item = self.state.items.get(item_id)
        if item is None:
            return None
        for detection in item.detections:
            if detection.id == region_id:
                return detection
        return None

    def _reference_source_detections(
        self,
        references: Iterable[dict[str, str]],
    ) -> list[DetectionRegion]:
        detections: list[DetectionRegion] = []
        for reference in references:
            detection = self._reference_source_detection(reference)
            if detection is not None:
                detections.append(detection)
        return detections

    def _is_distinct_reference_detection(
        self,
        detection: DetectionRegion,
        existing: Iterable[DetectionRegion],
    ) -> bool:
        existing_detections = list(existing)
        if not existing_detections:
            return True

        if detection.encoding:
            similarities = [
                self._cosine_similarity(detection.encoding, candidate.encoding)
                for candidate in existing_detections
                if candidate.encoding
            ]
            if similarities:
                return max(similarities) < REFERENCE_EMBEDDING_SIMILARITY_THRESHOLD

        if detection.signature:
            distances = [
                self._signature_distance(detection.signature, candidate.signature)
                for candidate in existing_detections
                if candidate.signature
            ]
            if distances:
                return min(distances) > REFERENCE_SIGNATURE_DISTANCE_THRESHOLD

        return True

    def _remove_reference_image_file(
        self,
        persona: Persona,
        path: Path,
        protected_paths: set[str] | None = None,
    ) -> None:
        if protected_paths and str(path) in protected_paths:
            return
        reference_root = self._reference_images_root() / persona.id
        try:
            resolved_root = reference_root.resolve()
            resolved_path = path.resolve()
            resolved_path.relative_to(resolved_root)
        except (OSError, ValueError):
            return
        try:
            resolved_path.unlink(missing_ok=True)
        except OSError:
            return

    def _extract_reference_crop(
        self,
        item: MediaItem,
        detection: DetectionRegion,
    ) -> Image.Image | None:
        source_image = self._load_reference_source_image(item, detection)
        if source_image is None:
            return None
        return self.vision.crop_region(source_image, detection.bbox)

    def _load_reference_source_image(
        self,
        item: MediaItem,
        detection: DetectionRegion,
    ) -> Image.Image | None:
        if detection.id.startswith("video-"):
            return self._load_video_reference_frame(item, detection.id)
        return self.vision.load_preview_image(self._spec_from_item(item))

    def _load_video_reference_frame(self, item: MediaItem, detection_id: str) -> Image.Image | None:
        if cv2 is None:
            return None
        match = re.match(r"^video-\d+-([0-9]+\.[0-9]+)-", detection_id)
        if not match:
            return None
        try:
            timestamp_seconds = float(match.group(1))
        except ValueError:
            return None

        video_path = self.vision.primary_video_path(self._spec_from_item(item))
        if not video_path:
            return None

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return None
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_frame)
        finally:
            capture.release()

    def _reference_images_root(self) -> Path:
        return self.config.cache_path.parent / "persona-references"

    def _cluster_previews_root(self) -> Path:
        return self.config.cache_path.parent / "cluster-previews"

    def _spec_from_item(self, item: MediaItem) -> MediaAssetSpec:
        return MediaAssetSpec(
            id=item.id,
            relative_key=item.relative_key,
            title=item.title,
            media_kind=item.media_kind,
            extension=item.extension,
            display_path=item.path,
            component_paths=list(item.component_paths),
            file_signature=item.file_signature,
            size_bytes=item.size_bytes,
            modified_ts=item.modified_ts,
        )

    def _is_pet_detection(self, detection: DetectionRegion) -> bool:
        return detection.kind.startswith("pet") or detection.label.lower() in PET_LABELS

    def _parse_query(self, query: str) -> tuple[dict[str, str], str]:
        field_filters: dict[str, str] = {}
        free_text_tokens: list[str] = []
        for token in query.split():
            if ":" in token:
                key, value = token.split(":", 1)
                if key in {"type", "tag", "person", "pet", "year"} and value:
                    field_filters[key] = value
                    continue
            free_text_tokens.append(token)
        return field_filters, " ".join(free_text_tokens).strip()

    def _search_items_from_store(
        self,
        *,
        query: str,
        media_kind: str,
        persona_kind: str,
        persona_id: str,
        favorites_only: bool,
        limit: int | None,
    ) -> list[MediaItem]:
        field_filters, free_text = self._parse_query(query)
        resolved_media_kind = field_filters.get("type", media_kind)
        resolved_tag = field_filters.get("tag", "").lower()
        resolved_year = field_filters.get("year", "")

        persona_ids: list[str] = []
        if persona_id:
            persona_ids = [persona_id]

        person_filter = field_filters.get("person", "").strip()
        pet_filter = field_filters.get("pet", "").strip()
        if person_filter:
            candidate_ids = self.store.find_persona_ids_by_name("person", person_filter)
            if not candidate_ids:
                return []
            persona_ids = candidate_ids if not persona_ids else [value for value in persona_ids if value in candidate_ids]
            if not persona_ids:
                return []
            persona_kind = "person"
        if pet_filter:
            candidate_ids = self.store.find_persona_ids_by_name("pet", pet_filter)
            if not candidate_ids:
                return []
            persona_ids = candidate_ids if not persona_ids else [value for value in persona_ids if value in candidate_ids]
            if not persona_ids:
                return []
            persona_kind = "pet"

        candidate_items = self.store.query_items(
            media_kind=resolved_media_kind,
            favorites_only=favorites_only,
            persona_ids=persona_ids or None,
            persona_kind=persona_kind,
            year=resolved_year,
            tag=resolved_tag,
            search_text=free_text,
            limit=SEARCH_CANDIDATE_LIMIT if free_text else limit,
        )
        if not free_text:
            return candidate_items

        scored_items: list[tuple[int, MediaItem]] = []
        for item in candidate_items:
            score = self._score_item(item, free_text)
            if score >= 35:
                scored_items.append((score, item))
        scored_items.sort(
            key=lambda entry: (entry[0], self._item_datetime(entry[1]), entry[1].modified_ts),
            reverse=True,
        )
        ranked_items = [item for _, item in scored_items]
        return ranked_items if limit is None else ranked_items[:limit]

    def _search_items_page_from_store(
        self,
        *,
        query: str,
        media_kind: str,
        persona_kind: str,
        persona_id: str,
        favorites_only: bool,
        offset: int,
        limit: int,
    ) -> MediaPage:
        field_filters, free_text = self._parse_query(query)
        resolved_media_kind = field_filters.get("type", media_kind)
        resolved_tag = field_filters.get("tag", "").lower()
        resolved_year = field_filters.get("year", "")

        persona_ids: list[str] = []
        if persona_id:
            persona_ids = [persona_id]

        person_filter = field_filters.get("person", "").strip()
        pet_filter = field_filters.get("pet", "").strip()
        if person_filter:
            candidate_ids = self.store.find_persona_ids_by_name("person", person_filter)
            if not candidate_ids:
                return MediaPage(items=[], has_more=False, next_offset=None)
            persona_ids = candidate_ids if not persona_ids else [value for value in persona_ids if value in candidate_ids]
            if not persona_ids:
                return MediaPage(items=[], has_more=False, next_offset=None)
            persona_kind = "person"
        if pet_filter:
            candidate_ids = self.store.find_persona_ids_by_name("pet", pet_filter)
            if not candidate_ids:
                return MediaPage(items=[], has_more=False, next_offset=None)
            persona_ids = candidate_ids if not persona_ids else [value for value in persona_ids if value in candidate_ids]
            if not persona_ids:
                return MediaPage(items=[], has_more=False, next_offset=None)
            persona_kind = "pet"

        if not free_text:
            candidate_items = self.store.query_items(
                media_kind=resolved_media_kind,
                favorites_only=favorites_only,
                persona_ids=persona_ids or None,
                persona_kind=persona_kind,
                year=resolved_year,
                tag=resolved_tag,
                search_text="",
                offset=offset,
                limit=limit + 1,
            )
            return self._page_from_items(candidate_items, offset=offset, limit=limit, pretrimmed=True)

        candidate_limit = max(SEARCH_CANDIDATE_LIMIT, offset + limit + 1)
        candidate_items = self.store.query_items(
            media_kind=resolved_media_kind,
            favorites_only=favorites_only,
            persona_ids=persona_ids or None,
            persona_kind=persona_kind,
            year=resolved_year,
            tag=resolved_tag,
            search_text=free_text,
            limit=candidate_limit,
        )

        scored_items: list[tuple[int, MediaItem]] = []
        for item in candidate_items:
            score = self._score_item(item, free_text)
            if score >= 35:
                scored_items.append((score, item))
        scored_items.sort(
            key=lambda entry: (entry[0], self._item_datetime(entry[1]), entry[1].modified_ts),
            reverse=True,
        )
        ranked_items = [item for _, item in scored_items]
        return self._page_from_items(ranked_items, offset=offset, limit=limit)

    def _apply_field_filters(
        self,
        items: list[MediaItem],
        field_filters: dict[str, str],
    ) -> list[MediaItem]:
        filtered = items
        if "type" in field_filters:
            filtered = [
                item for item in filtered if item.media_kind == field_filters["type"].lower()
            ]
        if "tag" in field_filters:
            tag = field_filters["tag"].lower()
            filtered = [item for item in filtered if tag in [entry.lower() for entry in item.tags]]
        if "year" in field_filters:
            filtered = [item for item in filtered if field_filters["year"] in item.captured_at]
        if "person" in field_filters:
            name = field_filters["person"].lower()
            filtered = [
                item
                for item in filtered
                if any(
                    persona.kind == "person" and persona.name.lower() == name
                    for persona in self.personas_for_item(item)
                )
            ]
        if "pet" in field_filters:
            name = field_filters["pet"].lower()
            filtered = [
                item
                for item in filtered
                if any(
                    persona.kind == "pet" and persona.name.lower() == name
                    for persona in self.personas_for_item(item)
                )
            ]
        return filtered

    def _score_item(self, item: MediaItem, query: str) -> int:
        blob = self._search_blob(item)
        if fuzz is not None:
            score = fuzz.token_set_ratio(query, blob)
            for token in query.split():
                score = max(score, fuzz.partial_ratio(token, blob))
            return int(score)

        score = int(difflib.SequenceMatcher(None, query, blob).ratio() * 100)
        for token in query.split():
            token_score = 100 if token in blob else int(difflib.SequenceMatcher(None, token, blob).ratio() * 100)
            score = max(score, token_score)
        return score

    def _search_blob(self, item: MediaItem) -> str:
        personas = " ".join(persona.name for persona in self.personas_for_item(item))
        detections = " ".join(region.label for region in item.detections)
        metadata_text = " ".join(f"{key} {value}" for key, value in item.metadata.items())
        parts = [
            item.title,
            item.relative_key,
            item.media_kind,
            " ".join(item.tags),
            personas,
            detections,
            metadata_text,
            item.notes,
        ]
        return " ".join(part.lower() for part in parts if part)

    def _guess_capture_date(self, spec: MediaAssetSpec) -> str:
        numeric_segments = [
            int(segment)
            for segment in re.findall(r"\d+", spec.relative_key)
            if segment.isdigit()
        ]
        year = next((value for value in numeric_segments if 1900 <= value <= 2100), None)
        if year:
            remaining = [value for value in numeric_segments if value != year]
            month = next((value for value in remaining if 1 <= value <= 12), 1)
            day = next((value for value in remaining if 1 <= value <= 31 and value != month), 1)
            try:
                return datetime(year, month, day, tzinfo=timezone.utc).isoformat()
            except ValueError:
                pass
        return datetime.fromtimestamp(spec.modified_ts, tz=timezone.utc).isoformat()

    def _parse_exif_datetime(self, value: object) -> str | None:
        if not value:
            return None
        try:
            parsed = datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
            return parsed.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            return None

    def _item_datetime(self, item: MediaItem) -> datetime:
        try:
            return datetime.fromisoformat(item.captured_at)
        except ValueError:
            return datetime.fromtimestamp(item.modified_ts, tz=timezone.utc)

    def _tags_from_relative_key(self, relative_key: str) -> list[str]:
        tokens = re.split(r"[^A-Za-z0-9]+", relative_key.lower())
        return [token for token in tokens if len(token) >= 3]

    def _euclidean_distance(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return math.inf
        total = 0.0
        for left_value, right_value in zip(left, right, strict=False):
            delta = left_value - right_value
            total += delta * delta
        return math.sqrt(total)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left:
            return -1.0
        numerator = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return -1.0
        return numerator / (left_norm * right_norm)

    def _signature_distance(self, left: str, right: str) -> float:
        if imagehash is not None:
            try:
                return float(imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right))
            except Exception:
                return math.inf
        try:
            return float((int(left, 16) ^ int(right, 16)).bit_count())
        except ValueError:
            if len(left) != len(right):
                return math.inf
            return float(sum(left_char != right_char for left_char, right_char in zip(left, right, strict=False)))

    def _scan_batch_size(self) -> int:
        configured = max(1, int(self.config.scan_batch_size))
        analysis_batch = max(1, int(self.config.analysis_batch_size))
        return max(configured, min(analysis_batch, 32))

    def _prefetch_workers(self) -> int:
        configured = max(1, int(self.config.prefetch_workers))
        cpu_count = os.cpu_count() or configured
        target = max(configured, min(8, cpu_count))
        return max(1, min(target, cpu_count))

    def _batch_detail(
        self,
        entries: list[tuple[str, MediaAssetSpec, MediaItem | None, str]],
    ) -> str:
        if not entries:
            return ""
        first = entries[0][1].relative_key
        last = entries[-1][1].relative_key
        if first == last:
            return first
        return f"{first} -> {last}"

    def _should_emit_scan_progress(self, completed: int, total: int) -> bool:
        if completed <= 1 or completed >= total:
            return True
        return completed % SCAN_PROGRESS_EMIT_INTERVAL == 0

    def _emit_progress(
        self,
        progress_callback: Callable[[ProgressUpdate], None] | None,
        update: ProgressUpdate,
    ) -> None:
        if progress_callback is not None:
            progress_callback(update)

    def _make_progress_update(
        self,
        *,
        phase: str,
        message: str,
        current: int = 0,
        total: int = 0,
        detail: str = "",
        indeterminate: bool = False,
        snapshot_ready: bool = False,
        overall_started_at: float | None = None,
        step_started_at: float | None = None,
    ) -> ProgressUpdate:
        now = monotonic()
        elapsed_seconds = None if overall_started_at is None else max(0.0, now - overall_started_at)
        step_seconds = None if step_started_at is None else max(0.0, now - step_started_at)
        eta_seconds = None
        if elapsed_seconds is not None:
            eta_seconds = self._estimate_eta_seconds(
                elapsed_seconds=elapsed_seconds,
                current=current,
                total=total,
            )
        return ProgressUpdate(
            phase=phase,
            message=message,
            current=current,
            total=total,
            detail=detail,
            indeterminate=indeterminate,
            snapshot_ready=snapshot_ready,
            elapsed_seconds=elapsed_seconds,
            eta_seconds=eta_seconds,
            step_seconds=step_seconds,
            timestamp_seconds=now,
        )

    def _estimate_eta_seconds(
        self,
        *,
        elapsed_seconds: float,
        current: int,
        total: int,
    ) -> float | None:
        if total <= 0 or current <= 0 or current >= total:
            return None
        work_rate = elapsed_seconds / current
        if work_rate <= 0:
            return None
        return max(0.0, work_rate * (total - current))

    def _sorted_asset_entries(
        self,
        assets: dict[str, MediaAssetSpec],
    ) -> list[tuple[str, MediaAssetSpec]]:
        return sorted(
            assets.items(),
            key=lambda entry: self._scan_sort_key(entry[1]),
            reverse=True,
        )

    def _scan_sort_key(
        self,
        spec: MediaAssetSpec,
    ) -> tuple[tuple[tuple[int, int, str], ...], float, str]:
        relative_path = Path(spec.relative_key)
        parent_parts = relative_path.parts[:-1]
        segment_key = tuple(self._segment_sort_key(part) for part in parent_parts)
        return (segment_key, spec.modified_ts, relative_path.name.lower())

    def _segment_sort_key(self, segment: str) -> tuple[int, int, str]:
        numbers = [int(value) for value in re.findall(r"\d+", segment)]
        if numbers:
            return (1, numbers[0], segment.lower())
        return (0, 0, segment.lower())
