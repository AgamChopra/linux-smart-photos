from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class DetectionRegion:
    id: str
    kind: str
    label: str
    confidence: float
    bbox: list[int]
    persona_id: str | None = None
    encoding: list[float] = field(default_factory=list)
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "persona_id": self.persona_id,
            "encoding": self.encoding,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionRegion":
        return cls(
            id=data["id"],
            kind=data["kind"],
            label=data["label"],
            confidence=float(data.get("confidence", 0.0)),
            bbox=[int(value) for value in data.get("bbox", [])],
            persona_id=data.get("persona_id"),
            encoding=[float(value) for value in data.get("encoding", [])],
            signature=data.get("signature"),
        )


@dataclass(slots=True)
class MediaItem:
    id: str
    path: str
    component_paths: list[str]
    relative_key: str
    title: str
    media_kind: str
    extension: str
    file_signature: str
    size_bytes: int
    modified_ts: float
    captured_at: str
    discovered_at: str
    thumbnail_path: str = ""
    width: int = 0
    height: int = 0
    duration_seconds: float = 0.0
    favorite: bool = False
    hidden: bool = False
    tags: list[str] = field(default_factory=list)
    detections: list[DetectionRegion] = field(default_factory=list)
    manual_persona_ids: list[str] = field(default_factory=list)
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "component_paths": self.component_paths,
            "relative_key": self.relative_key,
            "title": self.title,
            "media_kind": self.media_kind,
            "extension": self.extension,
            "file_signature": self.file_signature,
            "size_bytes": self.size_bytes,
            "modified_ts": self.modified_ts,
            "captured_at": self.captured_at,
            "discovered_at": self.discovered_at,
            "thumbnail_path": self.thumbnail_path,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "favorite": self.favorite,
            "hidden": self.hidden,
            "tags": self.tags,
            "detections": [entry.to_dict() for entry in self.detections],
            "manual_persona_ids": self.manual_persona_ids,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MediaItem":
        return cls(
            id=data["id"],
            path=data["path"],
            component_paths=list(data.get("component_paths", [])),
            relative_key=data.get("relative_key", data["path"]),
            title=data.get("title", ""),
            media_kind=data.get("media_kind", "image"),
            extension=data.get("extension", ""),
            file_signature=data.get("file_signature", ""),
            size_bytes=int(data.get("size_bytes", 0)),
            modified_ts=float(data.get("modified_ts", 0.0)),
            captured_at=data.get("captured_at", utc_now()),
            discovered_at=data.get("discovered_at", utc_now()),
            thumbnail_path=data.get("thumbnail_path", ""),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            favorite=bool(data.get("favorite", False)),
            hidden=bool(data.get("hidden", False)),
            tags=list(data.get("tags", [])),
            detections=[DetectionRegion.from_dict(entry) for entry in data.get("detections", [])],
            manual_persona_ids=list(data.get("manual_persona_ids", [])),
            notes=data.get("notes", ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class Persona:
    id: str
    name: str
    kind: str
    created_at: str
    color: str
    avatar_item_id: str = ""
    reference_encodings: list[list[float]] = field(default_factory=list)
    reference_signatures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "created_at": self.created_at,
            "color": self.color,
            "avatar_item_id": self.avatar_item_id,
            "reference_encodings": self.reference_encodings,
            "reference_signatures": self.reference_signatures,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Persona":
        return cls(
            id=data["id"],
            name=data["name"],
            kind=data.get("kind", "person"),
            created_at=data.get("created_at", utc_now()),
            color=data.get("color", "#4C7C9C"),
            avatar_item_id=data.get("avatar_item_id", ""),
            reference_encodings=[
                [float(value) for value in encoding]
                for encoding in data.get("reference_encodings", [])
            ],
            reference_signatures=list(data.get("reference_signatures", [])),
        )


@dataclass(slots=True)
class Album:
    id: str
    name: str
    created_at: str
    item_ids: list[str] = field(default_factory=list)
    description: str = ""
    query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "item_ids": self.item_ids,
            "description": self.description,
            "query": self.query,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Album":
        return cls(
            id=data["id"],
            name=data["name"],
            created_at=data.get("created_at", utc_now()),
            item_ids=list(data.get("item_ids", [])),
            description=data.get("description", ""),
            query=data.get("query", ""),
        )


@dataclass(slots=True)
class Memory:
    id: str
    title: str
    subtitle: str
    summary: str
    created_at: str
    memory_type: str
    item_ids: list[str] = field(default_factory=list)
    persona_ids: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "subtitle": self.subtitle,
            "summary": self.summary,
            "created_at": self.created_at,
            "memory_type": self.memory_type,
            "item_ids": self.item_ids,
            "persona_ids": self.persona_ids,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        return cls(
            id=data["id"],
            title=data["title"],
            subtitle=data.get("subtitle", ""),
            summary=data.get("summary", ""),
            created_at=data.get("created_at", utc_now()),
            memory_type=data.get("memory_type", "time"),
            item_ids=list(data.get("item_ids", [])),
            persona_ids=list(data.get("persona_ids", [])),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
        )


@dataclass(slots=True)
class LibraryState:
    schema_version: int = 1
    updated_at: str = field(default_factory=utc_now)
    items: dict[str, MediaItem] = field(default_factory=dict)
    personas: dict[str, Persona] = field(default_factory=dict)
    albums: dict[str, Album] = field(default_factory=dict)
    memories: dict[str, Memory] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "updated_at": self.updated_at,
            "items": {key: value.to_dict() for key, value in self.items.items()},
            "personas": {key: value.to_dict() for key, value in self.personas.items()},
            "albums": {key: value.to_dict() for key, value in self.albums.items()},
            "memories": {key: value.to_dict() for key, value in self.memories.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LibraryState":
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            updated_at=data.get("updated_at", utc_now()),
            items={
                key: MediaItem.from_dict(value)
                for key, value in data.get("items", {}).items()
            },
            personas={
                key: Persona.from_dict(value)
                for key, value in data.get("personas", {}).items()
            },
            albums={
                key: Album.from_dict(value)
                for key, value in data.get("albums", {}).items()
            },
            memories={
                key: Memory.from_dict(value)
                for key, value in data.get("memories", {}).items()
            },
        )
