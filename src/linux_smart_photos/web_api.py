from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import traceback
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from .branding import APP_DESCRIPTION, APP_NAME
from .config import AppConfig, config_file_path, load_config
from .models import MediaItem, Persona, utc_now
from .services.library import LibraryService, ProgressUpdate, SyncSummary, UnknownPersonaCluster


@dataclass(slots=True)
class JobRecord:
    id: str
    job_type: str
    status: str
    message: str
    current: int = 0
    total: int = 0
    detail: str = ""
    started_at: str = ""
    completed_at: str = ""
    error: str = ""
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BackgroundJobManager:
    def __init__(self, config: AppConfig, read_service: LibraryService, service_lock: threading.Lock) -> None:
        self.config = config
        self._read_service = read_service
        self._service_lock = service_lock
        self._lock = threading.Lock()
        self._current: JobRecord | None = None
        self._last: JobRecord | None = None

    def current_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if self._current and self._current.status == "running":
                return self._current.to_dict()
            if self._last:
                return self._last.to_dict()
        return None

    def start_sync(self, *, startup: bool = False) -> tuple[dict[str, Any], bool]:
        return self._start_job(
            job_type="startup-sync" if startup else "sync",
            initial_message="Preparing library sync",
            target=self._run_sync_job,
        )

    def start_model_download(self) -> tuple[dict[str, Any], bool]:
        return self._start_job(
            job_type="models",
            initial_message="Preparing AI model download",
            target=self._run_model_download_job,
        )

    def _start_job(
        self,
        *,
        job_type: str,
        initial_message: str,
        target: Callable[[LibraryService, Callable[[ProgressUpdate], None]], dict[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        with self._lock:
            if self._current and self._current.status == "running":
                return self._current.to_dict(), False

            record = JobRecord(
                id=uuid4().hex,
                job_type=job_type,
                status="running",
                message=initial_message,
                started_at=utc_now(),
            )
            self._current = record
            self._last = record

        thread = threading.Thread(
            target=self._execute_job,
            args=(record.id, target),
            daemon=True,
        )
        thread.start()
        return record.to_dict(), True

    def _execute_job(
        self,
        job_id: str,
        target: Callable[[LibraryService, Callable[[ProgressUpdate], None]], dict[str, Any]],
    ) -> None:
        worker_service = LibraryService(self.config)
        try:
            result = target(worker_service, lambda update: self._handle_progress(job_id, update))
        except Exception as exc:
            with self._lock:
                if self._current and self._current.id == job_id:
                    self._current.status = "failed"
                    self._current.error = f"{exc}"
                    self._current.message = "Background task failed"
                    self._current.completed_at = utc_now()
                    self._last = self._current
                    self._current = None
            return

        with self._service_lock:
            self._read_service.reload()

        with self._lock:
            record = self._last if self._last and self._last.id == job_id else None
            if self._current and self._current.id == job_id:
                record = self._current
            if record is None:
                return
            record.status = "completed"
            record.message = "Background task complete"
            record.result = result
            record.completed_at = utc_now()
            self._last = record
            self._current = None

    def _handle_progress(self, job_id: str, update: ProgressUpdate) -> None:
        with self._lock:
            if not self._current or self._current.id != job_id:
                return
            self._current.message = update.message
            self._current.current = update.current
            self._current.total = update.total
            self._current.detail = update.detail

        if update.snapshot_ready:
            with self._service_lock:
                self._read_service.reload()

    def _run_sync_job(
        self,
        service: LibraryService,
        progress_callback: Callable[[ProgressUpdate], None],
    ) -> dict[str, Any]:
        summary = service.sync(progress_callback=progress_callback)
        return {
            "added": summary.added,
            "updated": summary.updated,
            "removed": summary.removed,
        }

    def _run_model_download_job(
        self,
        service: LibraryService,
        progress_callback: Callable[[ProgressUpdate], None],
    ) -> dict[str, Any]:
        paths = service.download_recommended_models(progress_callback=progress_callback)
        return {"paths": paths}


class SmartPhotosApi:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.service = LibraryService(config)
        self.service_lock = threading.Lock()
        self.jobs = BackgroundJobManager(config, self.service, self.service_lock)

    def start_startup_sync(self) -> None:
        self.jobs.start_sync(startup=True)

    def status_payload(self) -> dict[str, Any]:
        with self.service_lock:
            return {
                "appName": APP_NAME,
                "appDescription": APP_DESCRIPTION,
                "mediaRoot": str(self.config.media_root_path),
                "databasePath": str(self.service.store.path),
                "cacheDir": str(self.config.cache_path),
                "updatedAt": self.service.store.updated_at(),
                "counts": {
                    "items": self.service.store.count_items(),
                    "personas": self.service.store.count_personas(),
                    "people": self.service.store.count_personas("person"),
                    "pets": self.service.store.count_personas("pet"),
                    "albums": self.service.store.count_albums(),
                    "memories": self.service.store.count_memories(),
                },
                "models": [self._serialize_model_status(status) for status in self.service.model_statuses()],
                "job": self.jobs.current_snapshot(),
            }

    def list_items(self, params: dict[str, list[str]]) -> list[dict[str, Any]]:
        query = self._single(params, "query")
        media_kind = self._single(params, "type", "all")
        persona_kind = self._single(params, "personaKind", "all")
        persona_id = self._single(params, "personaId")
        favorites_only = self._single(params, "favorites") == "1"
        limit = self._int_param(params, "limit", 180)
        with self.service_lock:
            items = self.service.search_items(
                query=query,
                media_kind=media_kind,
                persona_kind=persona_kind,
                persona_id=persona_id,
                favorites_only=favorites_only,
                limit=limit,
            )
            return [self._serialize_item_summary(item) for item in items]

    def item_payload(self, item_id: str) -> dict[str, Any] | None:
        with self.service_lock:
            item = self.service.store.load_item(item_id)
            if item is None:
                return None
            return self._serialize_item(item, include_relations=True)

    def list_personas(self, params: dict[str, list[str]]) -> list[dict[str, Any]]:
        kind = self._single(params, "kind", "all")
        with self.service_lock:
            personas = self.service.list_personas(kind=kind)
            return [self._serialize_persona(persona) for persona in personas]

    def persona_payload(self, persona_id: str) -> dict[str, Any] | None:
        with self.service_lock:
            persona = self.service.store.load_persona(persona_id)
            if persona is None:
                return None
            payload = self._serialize_persona(persona)
            payload["items"] = [
                self._serialize_item_summary(item)
                for item in self.service.items_for_persona(persona.id)[:180]
            ]
            payload["referenceImages"] = self.service.persona_reference_images(persona.id)
            return payload

    def list_albums(self) -> list[dict[str, Any]]:
        with self.service_lock:
            return [self._serialize_album(album) for album in self.service.list_albums()]

    def album_payload(self, album_id: str) -> dict[str, Any] | None:
        with self.service_lock:
            album = self.service.store.load_album(album_id)
            if album is None:
                return None
            payload = self._serialize_album(album)
            payload["items"] = [self._serialize_item_summary(item) for item in self.service.items_for_album(album.id)]
            return payload

    def list_memories(self) -> list[dict[str, Any]]:
        with self.service_lock:
            return [self._serialize_memory(memory) for memory in self.service.list_memories()]

    def memory_payload(self, memory_id: str) -> dict[str, Any] | None:
        with self.service_lock:
            memory = self.service.store.load_memory(memory_id)
            if memory is None:
                return None
            payload = self._serialize_memory(memory)
            payload["items"] = [self._serialize_item_summary(item) for item in self.service.items_for_memory(memory.id)]
            return payload

    def list_unknown_clusters(self, params: dict[str, list[str]]) -> list[dict[str, Any]]:
        kind = self._single(params, "kind", "all")
        with self.service_lock:
            clusters = self.service.list_unknown_persona_clusters(kind=kind)
            return [self._serialize_cluster(cluster) for cluster in clusters]

    def unknown_cluster_items(self, cluster_ids: list[str]) -> list[dict[str, Any]]:
        with self.service_lock:
            all_clusters = self.service.list_unknown_persona_clusters(kind="all")
            cluster_map = {cluster.id: cluster for cluster in all_clusters}
            selected = [cluster_map[cluster_id] for cluster_id in cluster_ids if cluster_id in cluster_map]
            items = self.service.items_for_unknown_clusters(selected)
            return [self._serialize_item_summary(item) for item in items]

    def create_persona(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.service_lock:
            persona = self.service.create_persona(
                str(payload.get("name", "")),
                str(payload.get("kind", "person")),
            )
            return self._serialize_persona(persona)

    def create_album(self, payload: dict[str, Any]) -> dict[str, Any]:
        item_ids = [str(item_id) for item_id in payload.get("itemIds", [])]
        with self.service_lock:
            album = self.service.create_album(str(payload.get("name", "")), item_ids)
            return self._serialize_album(album)

    def add_album_items(self, album_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        item_ids = [str(item_id) for item_id in payload.get("itemIds", [])]
        with self.service_lock:
            album = self.service.store.load_album(album_id)
            if album is None:
                return None
            self.service.add_items_to_album(album_id, item_ids)
            refreshed = self.service.store.load_album(album_id)
            return self._serialize_album(refreshed) if refreshed else None

    def delete_album(self, album_id: str) -> bool:
        with self.service_lock:
            album = self.service.store.load_album(album_id)
            if album is None:
                return False
            self.service.delete_album(album_id)
            return True

    def toggle_favorite(self, item_id: str) -> dict[str, Any] | None:
        with self.service_lock:
            if self.service.store.load_item(item_id) is None:
                return None
            self.service.toggle_favorite([item_id])
            refreshed = self.service.store.load_item(item_id)
            return self._serialize_item(refreshed, include_relations=True) if refreshed else None

    def assign_region(self, payload: dict[str, Any]) -> dict[str, Any]:
        item_id = str(payload.get("itemId", ""))
        region_id = str(payload.get("regionId", ""))
        with self.service_lock:
            persona = self.service.assign_region_to_persona(
                item_id,
                region_id,
                persona_id=str(payload.get("personaId", "")),
                new_name=str(payload.get("newName", "")),
                kind=str(payload.get("kind", "person")),
            )
            item = self.service.store.load_item(item_id)
            return {
                "persona": self._serialize_persona(persona),
                "item": self._serialize_item(item, include_relations=True) if item else None,
            }

    def clear_region(self, payload: dict[str, Any]) -> dict[str, Any]:
        item_id = str(payload.get("itemId", ""))
        region_id = str(payload.get("regionId", ""))
        with self.service_lock:
            self.service.clear_region_assignment(item_id, region_id)
            item = self.service.store.load_item(item_id)
            return {"item": self._serialize_item(item, include_relations=True) if item else None}

    def assign_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        item_id = str(payload.get("itemId", ""))
        with self.service_lock:
            persona = self.service.assign_item_to_persona(
                item_id,
                persona_id=str(payload.get("personaId", "")),
                new_name=str(payload.get("newName", "")),
                kind=str(payload.get("kind", "person")),
            )
            item = self.service.store.load_item(item_id)
            return {
                "persona": self._serialize_persona(persona),
                "item": self._serialize_item(item, include_relations=True) if item else None,
            }

    def clear_item_personas(self, payload: dict[str, Any]) -> dict[str, Any]:
        item_id = str(payload.get("itemId", ""))
        with self.service_lock:
            self.service.clear_item_personas(item_id)
            item = self.service.store.load_item(item_id)
            return {"item": self._serialize_item(item, include_relations=True) if item else None}

    def assign_unknown_clusters(self, payload: dict[str, Any]) -> dict[str, Any]:
        cluster_ids = [str(cluster_id) for cluster_id in payload.get("clusterIds", [])]
        with self.service_lock:
            cluster_map = {
                cluster.id: cluster
                for cluster in self.service.list_unknown_persona_clusters(kind="all")
            }
            clusters = [cluster_map[cluster_id] for cluster_id in cluster_ids if cluster_id in cluster_map]
            persona = self.service.assign_unknown_clusters_to_persona(
                clusters,
                persona_id=str(payload.get("personaId", "")),
                new_name=str(payload.get("newName", "")),
                kind=str(payload.get("kind", "person")),
            )
            return {"persona": self._serialize_persona(persona)}

    def start_sync(self) -> tuple[dict[str, Any], bool]:
        return self.jobs.start_sync()

    def start_model_download(self) -> tuple[dict[str, Any], bool]:
        return self.jobs.start_model_download()

    def _serialize_item_summary(self, item: MediaItem) -> dict[str, Any]:
        personas = self.service.personas_for_item(item)
        return {
            "id": item.id,
            "title": item.title,
            "path": item.path,
            "thumbnailPath": item.thumbnail_path,
            "mediaKind": item.media_kind,
            "capturedAt": item.captured_at,
            "favorite": item.favorite,
            "width": item.width,
            "height": item.height,
            "durationSeconds": item.duration_seconds,
            "tags": item.tags,
            "personaNames": [persona.name for persona in personas],
        }

    def _serialize_item(self, item: MediaItem, *, include_relations: bool = False) -> dict[str, Any]:
        payload = self._serialize_item_summary(item)
        payload.update(
            {
                "componentPaths": item.component_paths,
                "details": self.service.build_item_details(item),
                "detections": [
                    {
                        "id": detection.id,
                        "kind": detection.kind,
                        "label": detection.label,
                        "confidence": detection.confidence,
                        "bbox": detection.bbox,
                        "personaId": detection.persona_id,
                    }
                    for detection in item.detections
                ],
                "manualPersonaIds": item.manual_persona_ids,
                "notes": item.notes,
                "metadata": item.metadata,
            }
        )
        if include_relations:
            payload["personas"] = [
                {"id": persona.id, "name": persona.name, "kind": persona.kind}
                for persona in self.service.personas_for_item(item)
            ]
        return payload

    def _serialize_persona(self, persona: Persona) -> dict[str, Any]:
        avatar_thumbnail = ""
        if persona.avatar_item_id:
            avatar_item = self.service.store.load_item(persona.avatar_item_id)
            if avatar_item is not None:
                avatar_thumbnail = avatar_item.thumbnail_path
        return {
            "id": persona.id,
            "name": persona.name,
            "kind": persona.kind,
            "createdAt": persona.created_at,
            "color": persona.color,
            "avatarItemId": persona.avatar_item_id,
            "avatarThumbnailPath": avatar_thumbnail,
            "referenceImageCount": len(persona.reference_images),
        }

    def _serialize_album(self, album: Any) -> dict[str, Any]:
        return {
            "id": album.id,
            "name": album.name,
            "createdAt": album.created_at,
            "description": album.description,
            "itemCount": len(album.item_ids),
        }

    def _serialize_memory(self, memory: Any) -> dict[str, Any]:
        return {
            "id": memory.id,
            "title": memory.title,
            "subtitle": memory.subtitle,
            "summary": memory.summary,
            "createdAt": memory.created_at,
            "memoryType": memory.memory_type,
            "startDate": memory.start_date,
            "endDate": memory.end_date,
            "itemCount": len(memory.item_ids),
            "personaIds": memory.persona_ids,
        }

    def _serialize_cluster(self, cluster: UnknownPersonaCluster) -> dict[str, Any]:
        return {
            "id": cluster.id,
            "kind": cluster.kind,
            "label": cluster.label,
            "memberCount": cluster.member_count,
            "itemCount": cluster.item_count,
            "previewPath": cluster.preview_path,
            "latestCapturedAt": cluster.latest_captured_at,
            "averageConfidence": cluster.average_confidence,
        }

    def _serialize_model_status(self, status: Any) -> dict[str, Any]:
        return {
            "id": status.id,
            "title": status.title,
            "installed": status.installed,
            "localPath": status.local_path,
            "description": status.description,
        }

    def _single(self, params: dict[str, list[str]], key: str, default: str = "") -> str:
        values = params.get(key, [])
        return values[0] if values else default

    def _int_param(self, params: dict[str, list[str]], key: str, default: int) -> int:
        value = self._single(params, key)
        if not value:
            return default
        try:
            return max(1, int(value))
        except ValueError:
            return default


class SmartPhotosApiHandler(BaseHTTPRequestHandler):
    server: "SmartPhotosHttpServer"

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        try:
            self._dispatch_get()
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc), "trace": traceback.format_exc()})

    def do_POST(self) -> None:
        try:
            self._dispatch_post()
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc), "trace": traceback.format_exc()})

    def do_DELETE(self) -> None:
        try:
            self._dispatch_delete()
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc), "trace": traceback.format_exc()})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _dispatch_get(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        segments = self._segments(parsed.path)
        api = self.server.api

        if segments == ["api", "status"]:
            self._send_json(HTTPStatus.OK, api.status_payload())
            return
        if segments == ["api", "jobs", "current"]:
            self._send_json(HTTPStatus.OK, {"job": api.jobs.current_snapshot()})
            return
        if segments == ["api", "items"]:
            self._send_json(HTTPStatus.OK, {"items": api.list_items(params)})
            return
        if len(segments) == 3 and segments[:2] == ["api", "items"]:
            payload = api.item_payload(segments[2])
            if payload is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Item not found"})
                return
            self._send_json(HTTPStatus.OK, payload)
            return
        if segments == ["api", "personas"]:
            self._send_json(HTTPStatus.OK, {"personas": api.list_personas(params)})
            return
        if len(segments) == 3 and segments[:2] == ["api", "personas"]:
            payload = api.persona_payload(segments[2])
            if payload is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Persona not found"})
                return
            self._send_json(HTTPStatus.OK, payload)
            return
        if segments == ["api", "albums"]:
            self._send_json(HTTPStatus.OK, {"albums": api.list_albums()})
            return
        if len(segments) == 3 and segments[:2] == ["api", "albums"]:
            payload = api.album_payload(segments[2])
            if payload is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Album not found"})
                return
            self._send_json(HTTPStatus.OK, payload)
            return
        if segments == ["api", "memories"]:
            self._send_json(HTTPStatus.OK, {"memories": api.list_memories()})
            return
        if len(segments) == 3 and segments[:2] == ["api", "memories"]:
            payload = api.memory_payload(segments[2])
            if payload is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Memory not found"})
                return
            self._send_json(HTTPStatus.OK, payload)
            return
        if segments == ["api", "unknown-clusters"]:
            self._send_json(HTTPStatus.OK, {"clusters": api.list_unknown_clusters(params)})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found"})

    def _dispatch_post(self) -> None:
        parsed = urlparse(self.path)
        segments = self._segments(parsed.path)
        payload = self._read_json_body()
        api = self.server.api

        if segments == ["api", "jobs", "sync"]:
            job, started = api.start_sync()
            self._send_json(HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT, {"job": job, "started": started})
            return
        if segments == ["api", "jobs", "models", "recommended"]:
            job, started = api.start_model_download()
            self._send_json(HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT, {"job": job, "started": started})
            return
        if segments == ["api", "personas"]:
            self._send_json(HTTPStatus.CREATED, api.create_persona(payload))
            return
        if segments == ["api", "albums"]:
            self._send_json(HTTPStatus.CREATED, api.create_album(payload))
            return
        if len(segments) == 4 and segments[:2] == ["api", "albums"] and segments[3] == "items":
            album = api.add_album_items(segments[2], payload)
            if album is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Album not found"})
                return
            self._send_json(HTTPStatus.OK, album)
            return
        if len(segments) == 4 and segments[:2] == ["api", "items"] and segments[3] == "toggle-favorite":
            item = api.toggle_favorite(segments[2])
            if item is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Item not found"})
                return
            self._send_json(HTTPStatus.OK, item)
            return
        if segments == ["api", "corrections", "region", "assign"]:
            self._send_json(HTTPStatus.OK, api.assign_region(payload))
            return
        if segments == ["api", "corrections", "region", "clear"]:
            self._send_json(HTTPStatus.OK, api.clear_region(payload))
            return
        if segments == ["api", "corrections", "item", "assign"]:
            self._send_json(HTTPStatus.OK, api.assign_item(payload))
            return
        if segments == ["api", "corrections", "item", "clear"]:
            self._send_json(HTTPStatus.OK, api.clear_item_personas(payload))
            return
        if segments == ["api", "unknown-clusters", "items"]:
            cluster_ids = [str(cluster_id) for cluster_id in payload.get("clusterIds", [])]
            self._send_json(HTTPStatus.OK, {"items": api.unknown_cluster_items(cluster_ids)})
            return
        if segments == ["api", "unknown-clusters", "assign"]:
            self._send_json(HTTPStatus.OK, api.assign_unknown_clusters(payload))
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found"})

    def _dispatch_delete(self) -> None:
        parsed = urlparse(self.path)
        segments = self._segments(parsed.path)
        api = self.server.api

        if len(segments) == 3 and segments[:2] == ["api", "albums"]:
            deleted = api.delete_album(segments[2])
            if not deleted:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Album not found"})
                return
            self._send_json(HTTPStatus.OK, {"deleted": True})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found"})

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _segments(self, path: str) -> list[str]:
        return [segment for segment in path.strip("/").split("/") if segment]

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


class SmartPhotosHttpServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], api: SmartPhotosApi) -> None:
        super().__init__(server_address, SmartPhotosApiHandler)
        self.api = api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-photos-web-api",
        description=f"HTTP API server for {APP_NAME}.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Override the config file path.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Defaults to localhost.")
    parser.add_argument("--port", type=int, default=0, help="Bind port. Defaults to an ephemeral port.")
    parser.add_argument(
        "--startup-sync",
        action="store_true",
        help="Kick off a background sync as soon as the API starts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    api = SmartPhotosApi(config)
    server = SmartPhotosHttpServer((args.host, args.port), api)
    bound_host, bound_port = server.server_address
    print(f"SMART_PHOTOS_API_PORT={bound_port}", flush=True)
    print(f"SMART_PHOTOS_API_HOST={bound_host}", flush=True)
    if args.startup_sync:
        api.start_startup_sync()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
