from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import difflib
from itertools import combinations
import math
from pathlib import Path
import re
from typing import Iterable

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
from ..models import Album, DetectionRegion, MediaItem, Memory, Persona, utc_now
from ..store import JsonLibraryStore
from .model_manager import ModelStatus
from .vision import CAT_LABELS, PET_LABELS, VisionAnalyzer


PERSONA_COLORS = [
    "#0D3B66",
    "#3A7D44",
    "#A23B72",
    "#B56576",
    "#33658A",
    "#6D597A",
    "#5E6472",
]


@dataclass(slots=True)
class SyncSummary:
    added: int
    updated: int
    removed: int


class LibraryService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.store = JsonLibraryStore(config.database_file)
        self.state = self.store.load()
        self.vision = VisionAnalyzer(config)
        self.model_manager = self.vision.model_manager
        config.cache_path.mkdir(parents=True, exist_ok=True)

    def sync(self) -> SyncSummary:
        assets = build_asset_specs(self.config.media_root_path)
        existing_ids = set(self.state.items)
        current_ids = set(assets)
        removed_ids = existing_ids - current_ids
        added = 0
        updated = 0

        for item_id in removed_ids:
            self.state.items.pop(item_id, None)

        for item_id, spec in assets.items():
            existing = self.state.items.get(item_id)
            if existing and existing.file_signature == spec.file_signature:
                continue

            self.state.items[item_id] = self._build_item(spec, existing)
            if existing is None:
                added += 1
            else:
                updated += 1

        self._cleanup_collections()
        self.regenerate_memories()
        self.save()
        return SyncSummary(added=added, updated=updated, removed=len(removed_ids))

    def save(self) -> None:
        self.state.updated_at = utc_now()
        self.store.save(self.state)

    def list_items(self) -> list[MediaItem]:
        return sorted(
            self.state.items.values(),
            key=lambda item: (self._item_datetime(item), item.modified_ts),
            reverse=True,
        )

    def list_personas(self, kind: str = "all") -> list[Persona]:
        personas = list(self.state.personas.values())
        if kind != "all":
            personas = [persona for persona in personas if persona.kind == kind]
        return sorted(personas, key=lambda persona: (persona.kind, persona.name.lower()))

    def list_albums(self) -> list[Album]:
        return sorted(self.state.albums.values(), key=lambda album: album.name.lower())

    def list_memories(self) -> list[Memory]:
        return sorted(
            self.state.memories.values(),
            key=lambda memory: memory.end_date or memory.created_at,
            reverse=True,
        )

    def model_statuses(self) -> list[ModelStatus]:
        return self.model_manager.all_statuses()

    def download_recommended_models(self) -> list[str]:
        paths = self.model_manager.download_recommended_models()
        self.vision = VisionAnalyzer(self.config)
        self.model_manager = self.vision.model_manager
        return paths

    def download_model(self, model_id: str) -> str:
        path = self.model_manager.download_model(model_id)
        self.vision = VisionAnalyzer(self.config)
        self.model_manager = self.vision.model_manager
        return path

    def search_items(
        self,
        query: str = "",
        media_kind: str = "all",
        persona_kind: str = "all",
        persona_id: str = "",
        favorites_only: bool = False,
    ) -> list[MediaItem]:
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
            return items

        scored_items: list[tuple[int, MediaItem]] = []
        for item in items:
            score = self._score_item(item, free_text)
            if score >= 35:
                scored_items.append((score, item))
        scored_items.sort(
            key=lambda entry: (entry[0], self._item_datetime(entry[1]), entry[1].modified_ts),
            reverse=True,
        )
        return [item for _, item in scored_items]

    def items_for_persona(self, persona_id: str) -> list[MediaItem]:
        return [
            item
            for item in self.list_items()
            if persona_id in self.item_persona_ids(item)
        ]

    def items_for_album(self, album_id: str) -> list[MediaItem]:
        album = self.state.albums.get(album_id)
        if not album:
            return []
        return [self.state.items[item_id] for item_id in album.item_ids if item_id in self.state.items]

    def items_for_memory(self, memory_id: str) -> list[MediaItem]:
        memory = self.state.memories.get(memory_id)
        if not memory:
            return []
        return [self.state.items[item_id] for item_id in memory.item_ids if item_id in self.state.items]

    def item_persona_ids(self, item: MediaItem) -> list[str]:
        persona_ids = set(item.manual_persona_ids)
        for detection in item.detections:
            if detection.persona_id:
                persona_ids.add(detection.persona_id)
        return sorted(persona_ids)

    def personas_for_item(self, item: MediaItem) -> list[Persona]:
        personas = []
        for persona_id in self.item_persona_ids(item):
            persona = self.state.personas.get(persona_id)
            if persona:
                personas.append(persona)
        return sorted(personas, key=lambda persona: persona.name.lower())

    def build_item_details(self, item: MediaItem) -> str:
        personas = ", ".join(persona.name for persona in self.personas_for_item(item)) or "None"
        tags = ", ".join(item.tags[:24]) or "None"
        lines = [
            f"Title: {item.title}",
            f"Type: {item.media_kind}",
            f"Path: {item.path}",
            f"Captured: {item.captured_at}",
            f"Size: {item.width} x {item.height}",
            f"Duration: {item.duration_seconds:.1f}s" if item.duration_seconds else "Duration: n/a",
            f"Favorite: {'yes' if item.favorite else 'no'}",
            f"People/Pets: {personas}",
            f"Tags: {tags}",
        ]
        video_ai_frames = item.metadata.get("video_ai_frames_analyzed")
        if video_ai_frames:
            lines.append(f"Video AI frames: {video_ai_frames}")
        if item.component_paths and len(item.component_paths) > 1:
            lines.append(f"Components: {len(item.component_paths)}")
        return "\n".join(lines)

    def create_persona(self, name: str, kind: str) -> Persona:
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
        persona = self._resolve_persona(persona_id, new_name, kind)
        item = self.state.items[item_id]
        for detection in item.detections:
            if detection.id != region_id:
                continue
            detection.persona_id = persona.id
            if detection.encoding:
                self._remember_embedding(persona, detection.encoding)
            if detection.signature:
                self._remember_signature(persona, detection.signature)
            if not persona.avatar_item_id:
                persona.avatar_item_id = item.id
            break
        self.regenerate_memories()
        self.save()
        return persona

    def clear_region_assignment(self, item_id: str, region_id: str) -> None:
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
        persona = self._resolve_persona(persona_id, new_name, kind)
        item = self.state.items[item_id]
        if persona.id not in item.manual_persona_ids:
            item.manual_persona_ids.append(persona.id)
        if persona.kind == "person":
            face_detections = [entry for entry in item.detections if entry.kind == "face" and entry.encoding]
            if len(face_detections) == 1:
                self._remember_embedding(persona, face_detections[0].encoding)
        if persona.kind == "pet":
            for detection in item.detections:
                if self._is_pet_detection(detection) and detection.encoding:
                    self._remember_embedding(persona, detection.encoding)
                if detection.signature and self._is_pet_detection(detection):
                    self._remember_signature(persona, detection.signature)
        if not persona.avatar_item_id:
            persona.avatar_item_id = item.id
        self.regenerate_memories()
        self.save()
        return persona

    def clear_item_personas(self, item_id: str) -> None:
        item = self.state.items[item_id]
        item.manual_persona_ids = []
        self.regenerate_memories()
        self.save()

    def create_album(self, name: str, item_ids: Iterable[str] = ()) -> Album:
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
        album = self.state.albums[album_id]
        for item_id in item_ids:
            if item_id in self.state.items and item_id not in album.item_ids:
                album.item_ids.append(item_id)
        self.save()

    def delete_album(self, album_id: str) -> None:
        self.state.albums.pop(album_id, None)
        self.save()

    def toggle_favorite(self, item_ids: Iterable[str]) -> bool:
        selected_items = [self.state.items[item_id] for item_id in item_ids if item_id in self.state.items]
        if not selected_items:
            return False
        new_state = not all(item.favorite for item in selected_items)
        for item in selected_items:
            item.favorite = new_state
        self.save()
        return new_state

    def regenerate_memories(self) -> None:
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
        metadata = self._extract_metadata(spec)
        thumbnail_path = self._ensure_thumbnail(spec)
        analysis = self.vision.analyze(spec)
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
            thumbnail_path=thumbnail_path,
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

        image.thumbnail((self.config.thumbnail_size, self.config.thumbnail_size))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(cache_path, format="JPEG", quality=88)
        return str(cache_path)

    def _merge_detection_assignments(
        self,
        item: MediaItem,
        existing: MediaItem | None,
    ) -> None:
        if not existing:
            return
        existing_regions = {region.id: region for region in existing.detections}
        for detection in item.detections:
            prior = existing_regions.get(detection.id)
            if prior and prior.persona_id in self.state.personas:
                detection.persona_id = prior.persona_id

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

    def _cleanup_collections(self) -> None:
        live_item_ids = set(self.state.items)
        for album in self.state.albums.values():
            album.item_ids = [item_id for item_id in album.item_ids if item_id in live_item_ids]
        for memory in self.state.memories.values():
            memory.item_ids = [item_id for item_id in memory.item_ids if item_id in live_item_ids]
        for persona in self.state.personas.values():
            if persona.avatar_item_id and persona.avatar_item_id not in live_item_ids:
                persona.avatar_item_id = ""

    def _resolve_persona(self, persona_id: str, new_name: str, kind: str) -> Persona:
        if persona_id:
            persona = self.state.personas.get(persona_id)
            if not persona:
                raise ValueError("Persona not found.")
            return persona
        return self.create_persona(new_name, kind)

    def _remember_embedding(self, persona: Persona, encoding: list[float]) -> None:
        if not persona.reference_encodings:
            persona.reference_encodings.append(encoding)
            return
        closest = max(
            self._cosine_similarity(encoding, candidate)
            for candidate in persona.reference_encodings
        )
        if closest < 0.985:
            persona.reference_encodings.append(encoding)

    def _remember_signature(self, persona: Persona, signature: str) -> None:
        if signature not in persona.reference_signatures:
            persona.reference_signatures.append(signature)

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
