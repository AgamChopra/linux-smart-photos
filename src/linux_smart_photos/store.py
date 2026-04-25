from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Sequence

from .models import Album, DetectionRecord, LibraryState, MediaItem, Memory, Persona


SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


class JsonLibraryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> LibraryState:
        if not self.path.exists():
            return LibraryState()

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return LibraryState.from_dict(payload)

    def save(self, state: LibraryState) -> None:
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(state.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self.path)


class SQLiteLibraryStore:
    def __init__(self, path: Path) -> None:
        self.requested_path = path
        self.path = self._sqlite_path_for(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.migrated_from_legacy_json = self._migrate_legacy_json_if_needed()
        self._backfill_detections_if_needed()

    def schema_version(self) -> int:
        with self._connect() as conn:
            return int(self._metadata_value(conn, "schema_version") or 1)

    def updated_at(self) -> str:
        with self._connect() as conn:
            return self._metadata_value(conn, "updated_at") or LibraryState().updated_at

    def has_state(self) -> bool:
        return self._has_state()

    def count_items(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM items").fetchone()
        return int(row["count"]) if row is not None else 0

    def count_personas(self, kind: str = "all") -> int:
        with self._connect() as conn:
            if kind == "all":
                row = conn.execute("SELECT COUNT(*) AS count FROM personas").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS count FROM personas WHERE kind = ?",
                    (kind,),
                ).fetchone()
        return int(row["count"]) if row is not None else 0

    def count_albums(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM albums").fetchone()
        return int(row["count"]) if row is not None else 0

    def count_memories(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM memories").fetchone()
        return int(row["count"]) if row is not None else 0

    def legacy_json_candidates(self) -> list[Path]:
        return [candidate for candidate in self._legacy_json_paths_for(self.requested_path) if candidate.exists()]

    def delete_legacy_json_files(self) -> list[Path]:
        deleted_paths: list[Path] = []
        has_state = self._has_state()
        for candidate in self.legacy_json_candidates():
            if not has_state:
                try:
                    legacy_state = JsonLibraryStore(candidate).load()
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    legacy_state.items
                    or legacy_state.personas
                    or legacy_state.albums
                    or legacy_state.memories
                ):
                    continue
            try:
                candidate.unlink()
            except FileNotFoundError:
                continue
            deleted_paths.append(candidate)
        return deleted_paths

    def load(self) -> LibraryState:
        with self._connect() as conn:
            schema_version = int(self._metadata_value(conn, "schema_version") or 1)
            updated_at = self._metadata_value(conn, "updated_at") or LibraryState().updated_at
            items = {
                row["id"]: MediaItem.from_dict(json.loads(row["payload"]))
                for row in conn.execute("SELECT id, payload FROM items")
            }
            personas = {
                row["id"]: Persona.from_dict(json.loads(row["payload"]))
                for row in conn.execute("SELECT id, payload FROM personas")
            }
            albums = {
                row["id"]: Album.from_dict(json.loads(row["payload"]))
                for row in conn.execute("SELECT id, payload FROM albums")
            }
            memories = {
                row["id"]: Memory.from_dict(json.loads(row["payload"]))
                for row in conn.execute("SELECT id, payload FROM memories")
            }
        return LibraryState(
            schema_version=schema_version,
            updated_at=updated_at,
            items=items,
            personas=personas,
            albums=albums,
            memories=memories,
        )

    def save(self, state: LibraryState) -> None:
        with self._connect() as conn:
            self._set_metadata(conn, "schema_version", str(state.schema_version))
            self._set_metadata(conn, "updated_at", state.updated_at)
            self._replace_entity_table(conn, "personas", (self._persona_row(persona) for persona in state.personas.values()))
            self._replace_entity_table(conn, "albums", (self._album_row(album) for album in state.albums.values()))
            self._replace_entity_table(conn, "memories", (self._memory_row(memory) for memory in state.memories.values()))
            self._replace_entity_table(conn, "items", (self._item_row(item) for item in state.items.values()))
            self._rebuild_item_indexes(conn, state.items.values(), state.personas)
            self._rebuild_detection_rows(conn, state.items.values(), updated_at=state.updated_at)
            self._delete_empty_unknown_clusters(conn)

    def save_items_progress(
        self,
        items: Iterable[MediaItem],
        removed_item_ids: Iterable[str],
        *,
        updated_at: str,
        schema_version: int,
        personas: dict[str, Persona] | None = None,
    ) -> None:
        buffered_items = list(items)
        removed_ids = [item_id for item_id in removed_item_ids if item_id]
        with self._connect() as conn:
            self._set_metadata(conn, "schema_version", str(schema_version))
            self._set_metadata(conn, "updated_at", updated_at)
            if removed_ids:
                conn.executemany("DELETE FROM items WHERE id = ?", ((item_id,) for item_id in removed_ids))
                self._delete_item_indexes(conn, removed_ids)
                self._touch_cluster_revision(conn, updated_at)
            if buffered_items:
                conn.executemany(
                    """
                    INSERT INTO items (
                      id, title, media_kind, captured_at, modified_ts, favorite, hidden, path, payload
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      title=excluded.title,
                      media_kind=excluded.media_kind,
                      captured_at=excluded.captured_at,
                      modified_ts=excluded.modified_ts,
                      favorite=excluded.favorite,
                      hidden=excluded.hidden,
                      path=excluded.path,
                      payload=excluded.payload
                    """,
                    [self._item_row(item) for item in buffered_items],
                )
                self._delete_item_indexes(conn, [item.id for item in buffered_items])
                self._insert_item_indexes(conn, buffered_items, personas or {})
                self._replace_item_detection_rows(conn, buffered_items, updated_at=updated_at)
            self._delete_empty_unknown_clusters(conn)

    def load_item(self, item_id: str) -> MediaItem | None:
        row = self._fetch_payload_row("items", item_id)
        return None if row is None else MediaItem.from_dict(json.loads(row["payload"]))

    def load_items_by_ids(self, item_ids: Sequence[str]) -> list[MediaItem]:
        ordered_ids = [item_id for item_id in item_ids if item_id]
        if not ordered_ids:
            return []
        placeholders = ",".join("?" for _ in ordered_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id, payload FROM items WHERE id IN ({placeholders})",
                ordered_ids,
            ).fetchall()
        items_by_id = {
            row["id"]: MediaItem.from_dict(json.loads(row["payload"]))
            for row in rows
        }
        return [items_by_id[item_id] for item_id in ordered_ids if item_id in items_by_id]

    def query_items_by_ids(
        self,
        item_ids: Sequence[str],
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[MediaItem]:
        filtered_ids = [item_id for item_id in item_ids if item_id]
        if not filtered_ids:
            return []
        placeholders = ",".join("?" for _ in filtered_ids)
        params: list[Any] = list(filtered_ids)
        sql = (
            f"SELECT payload FROM items WHERE id IN ({placeholders}) "
            "ORDER BY captured_at DESC, modified_ts DESC"
        )
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
            if offset > 0:
                sql += " OFFSET ?"
                params.append(offset)
        elif offset > 0:
            sql += " LIMIT -1 OFFSET ?"
            params.append(offset)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [MediaItem.from_dict(json.loads(row["payload"])) for row in rows]

    def list_personas(self, kind: str = "all") -> list[Persona]:
        params: list[Any] = []
        sql = "SELECT payload FROM personas"
        if kind != "all":
            sql += " WHERE kind = ?"
            params.append(kind)
        sql += " ORDER BY kind ASC, name COLLATE NOCASE ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Persona.from_dict(json.loads(row["payload"])) for row in rows]

    def load_persona(self, persona_id: str) -> Persona | None:
        row = self._fetch_payload_row("personas", persona_id)
        return None if row is None else Persona.from_dict(json.loads(row["payload"]))

    def load_personas_by_ids(self, persona_ids: Sequence[str]) -> list[Persona]:
        ordered_ids = [persona_id for persona_id in persona_ids if persona_id]
        if not ordered_ids:
            return []
        placeholders = ",".join("?" for _ in ordered_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id, payload FROM personas WHERE id IN ({placeholders})",
                ordered_ids,
            ).fetchall()
        personas_by_id = {
            row["id"]: Persona.from_dict(json.loads(row["payload"]))
            for row in rows
        }
        return [personas_by_id[persona_id] for persona_id in ordered_ids if persona_id in personas_by_id]

    def find_persona_ids_by_name(self, kind: str, name: str) -> list[str]:
        normalized = name.strip().lower()
        if not normalized:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id FROM personas
                WHERE kind = ? AND lower(name) = ?
                ORDER BY name COLLATE NOCASE ASC
                """,
                (kind, normalized),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def list_albums(self) -> list[Album]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM albums ORDER BY name COLLATE NOCASE ASC").fetchall()
        return [Album.from_dict(json.loads(row["payload"])) for row in rows]

    def load_album(self, album_id: str) -> Album | None:
        row = self._fetch_payload_row("albums", album_id)
        return None if row is None else Album.from_dict(json.loads(row["payload"]))

    def list_memories(self) -> list[Memory]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM memories
                ORDER BY
                  CASE WHEN end_date = '' THEN created_at ELSE end_date END DESC,
                  created_at DESC
                """
            ).fetchall()
        return [Memory.from_dict(json.loads(row["payload"])) for row in rows]

    def load_memory(self, memory_id: str) -> Memory | None:
        row = self._fetch_payload_row("memories", memory_id)
        return None if row is None else Memory.from_dict(json.loads(row["payload"]))

    def query_items(
        self,
        *,
        media_kind: str = "all",
        favorites_only: bool = False,
        include_hidden: bool = False,
        persona_ids: Sequence[str] | None = None,
        persona_kind: str = "all",
        year: str = "",
        tag: str = "",
        search_text: str = "",
        offset: int = 0,
        limit: int | None = None,
    ) -> list[MediaItem]:
        joins: list[str] = []
        clauses: list[str] = []
        params: list[Any] = []

        if not include_hidden:
            clauses.append("i.hidden = 0")
        if media_kind != "all":
            clauses.append("i.media_kind = ?")
            params.append(media_kind)
        if favorites_only:
            clauses.append("i.favorite = 1")
        if year:
            clauses.append("substr(i.captured_at, 1, 4) = ?")
            params.append(year)
        if tag:
            joins.append("JOIN item_tags it ON it.item_id = i.id")
            clauses.append("it.tag = ?")
            params.append(tag.lower())
        if persona_ids:
            joins.append("JOIN item_personas ip ON ip.item_id = i.id")
            placeholders = ",".join("?" for _ in persona_ids)
            clauses.append(f"ip.persona_id IN ({placeholders})")
            params.extend(persona_ids)
        elif persona_kind != "all":
            joins.append("JOIN item_personas ip ON ip.item_id = i.id")
            joins.append("JOIN personas p ON p.id = ip.persona_id")
            clauses.append("p.kind = ?")
            params.append(persona_kind)

        search_tokens = self._search_tokens(search_text)
        if search_tokens:
            joins.append("JOIN item_search sx ON sx.item_id = i.id")
            for token in search_tokens:
                clauses.append("sx.search_blob LIKE ?")
                params.append(f"%{token}%")

        sql = "SELECT DISTINCT i.id FROM items i"
        if joins:
            sql += " " + " ".join(dict.fromkeys(joins))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY i.captured_at DESC, i.modified_ts DESC"
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        if limit is not None and limit > 0:
            if offset > 0:
                sql += " OFFSET ?"
                params.append(offset)
        elif offset > 0:
            sql += " LIMIT -1 OFFSET ?"
            params.append(offset)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        item_ids = [str(row["id"]) for row in rows]
        return self.load_items_by_ids(item_ids)

    def query_detections(
        self,
        *,
        cluster_kind: str = "all",
        dirty_only: bool = False,
        item_ids: Sequence[str] | None = None,
    ) -> list[DetectionRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if cluster_kind == "person":
            clauses.append("kind = 'face'")
        elif cluster_kind == "pet":
            clauses.append("kind IN ('pet', 'pet_face')")
        elif cluster_kind != "all":
            clauses.append("kind = ?")
            params.append(cluster_kind)
        if dirty_only:
            clauses.append("cluster_dirty = 1")
        if item_ids:
            filtered_ids = [item_id for item_id in item_ids if item_id]
            if not filtered_ids:
                return []
            placeholders = ",".join("?" for _ in filtered_ids)
            clauses.append(f"item_id IN ({placeholders})")
            params.extend(filtered_ids)

        sql = (
            "SELECT item_id, detection_id, kind, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, "
            "persona_id, encoding_json, signature, source_frame_token, captured_at, pipeline_revision, "
            "cluster_dirty, cluster_dirty_revision, updated_at "
            "FROM detections"
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY captured_at DESC, item_id ASC, detection_id ASC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_detection_record(row) for row in rows]

    def list_unknown_clusters(self, kind: str = "all") -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if kind != "all":
            clauses.append("uc.kind = ?")
            params.append(kind)

        sql = (
            "SELECT "
            "  uc.cluster_id, uc.kind, uc.label, uc.member_count, uc.item_count, "
            "  uc.representative_detection_id, uc.representative_item_id, uc.preview_path, "
            "  uc.latest_captured_at, uc.average_confidence, uc.revision, uc.is_partial, uc.updated_at, "
            "  ucm.item_id AS member_item_id, ucm.detection_id AS member_detection_id "
            "FROM unknown_clusters uc "
            "LEFT JOIN unknown_cluster_members ucm ON ucm.cluster_id = uc.cluster_id"
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY uc.member_count DESC, uc.item_count DESC, uc.latest_captured_at DESC, uc.cluster_id ASC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        clusters_by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            cluster_id = str(row["cluster_id"])
            cluster = clusters_by_id.get(cluster_id)
            if cluster is None:
                cluster = {
                    "id": cluster_id,
                    "kind": str(row["kind"]),
                    "label": str(row["label"]),
                    "member_count": int(row["member_count"]),
                    "item_count": int(row["item_count"]),
                    "representative_detection_id": str(row["representative_detection_id"]),
                    "representative_item_id": str(row["representative_item_id"]),
                    "preview_path": str(row["preview_path"]),
                    "latest_captured_at": str(row["latest_captured_at"]),
                    "average_confidence": float(row["average_confidence"]),
                    "revision": str(row["revision"]),
                    "is_partial": bool(int(row["is_partial"])),
                    "updated_at": str(row["updated_at"]),
                    "member_ids": [],
                    "item_ids": set(),
                }
                clusters_by_id[cluster_id] = cluster
            member_item_id = row["member_item_id"]
            member_detection_id = row["member_detection_id"]
            if member_item_id is None or member_detection_id is None:
                continue
            member = (str(member_item_id), str(member_detection_id))
            cluster["member_ids"].append(member)
            cluster["item_ids"].add(member[0])

        results: list[dict[str, Any]] = []
        for cluster in clusters_by_id.values():
            member_ids = sorted(
                {(item_id, detection_id) for item_id, detection_id in cluster["member_ids"]},
                key=lambda entry: f"{entry[0]}:{entry[1]}",
            )
            if not member_ids:
                continue
            item_ids = sorted(cluster["item_ids"])
            cluster["member_ids"] = member_ids
            cluster["item_ids"] = item_ids
            cluster["member_count"] = len(member_ids)
            cluster["item_count"] = len(item_ids)
            results.append(cluster)
        results.sort(
            key=lambda cluster: (
                int(cluster["member_count"]),
                int(cluster["item_count"]),
                str(cluster["latest_captured_at"]),
            ),
            reverse=True,
        )
        return results

    def load_unknown_cluster_states(self, kind: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  uc.cluster_id,
                  uc.kind,
                  uc.label,
                  uc.representative_detection_id,
                  uc.representative_item_id,
                  uc.preview_path,
                  uc.latest_captured_at,
                  uc.average_confidence,
                  uc.revision,
                  uc.is_partial,
                  uc.updated_at,
                  d.item_id,
                  d.detection_id,
                  d.kind AS detection_kind,
                  d.label AS detection_label,
                  d.confidence,
                  d.bbox_x,
                  d.bbox_y,
                  d.bbox_w,
                  d.bbox_h,
                  d.persona_id,
                  d.encoding_json,
                  d.signature,
                  d.source_frame_token,
                  d.captured_at,
                  d.pipeline_revision,
                  d.cluster_dirty,
                  d.cluster_dirty_revision,
                  d.updated_at AS detection_updated_at
                FROM unknown_clusters uc
                JOIN unknown_cluster_members ucm
                  ON ucm.cluster_id = uc.cluster_id
                JOIN detections d
                  ON d.item_id = ucm.item_id
                 AND d.detection_id = ucm.detection_id
                WHERE uc.kind = ?
                ORDER BY uc.cluster_id ASC, d.captured_at DESC, d.item_id ASC, d.detection_id ASC
                """,
                (kind,),
            ).fetchall()

        clusters_by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            cluster_id = str(row["cluster_id"])
            cluster = clusters_by_id.get(cluster_id)
            if cluster is None:
                cluster = {
                    "id": cluster_id,
                    "kind": str(row["kind"]),
                    "label": str(row["label"]),
                    "representative_detection_id": str(row["representative_detection_id"]),
                    "representative_item_id": str(row["representative_item_id"]),
                    "preview_path": str(row["preview_path"]),
                    "latest_captured_at": str(row["latest_captured_at"]),
                    "average_confidence": float(row["average_confidence"]),
                    "revision": str(row["revision"]),
                    "is_partial": bool(int(row["is_partial"])),
                    "updated_at": str(row["updated_at"]),
                    "detections": [],
                }
                clusters_by_id[cluster_id] = cluster
            cluster["detections"].append(self._row_to_detection_record(row))
        return list(clusters_by_id.values())

    def replace_unknown_clusters(
        self,
        kind: str,
        clusters: Sequence[dict[str, Any]],
        *,
        revision: str,
        partial: bool,
    ) -> None:
        with self._connect() as conn:
            existing_cluster_ids = [
                str(row["cluster_id"])
                for row in conn.execute(
                    "SELECT cluster_id FROM unknown_clusters WHERE kind = ?",
                    (kind,),
                ).fetchall()
            ]
            if existing_cluster_ids:
                placeholders = ",".join("?" for _ in existing_cluster_ids)
                conn.execute(
                    f"DELETE FROM unknown_cluster_members WHERE cluster_id IN ({placeholders})",
                    existing_cluster_ids,
                )
                conn.execute(
                    f"DELETE FROM unknown_clusters WHERE cluster_id IN ({placeholders})",
                    existing_cluster_ids,
                )

            if not clusters:
                return

            cluster_rows = [
                (
                    str(cluster["id"]),
                    str(cluster["kind"]),
                    str(cluster["label"]),
                    int(cluster["member_count"]),
                    int(cluster["item_count"]),
                    str(cluster["representative_detection_id"]),
                    str(cluster["representative_item_id"]),
                    str(cluster["preview_path"]),
                    str(cluster["latest_captured_at"]),
                    float(cluster["average_confidence"]),
                    revision,
                    int(partial),
                    revision,
                )
                for cluster in clusters
            ]
            conn.executemany(
                """
                INSERT INTO unknown_clusters (
                  cluster_id, kind, label, member_count, item_count,
                  representative_detection_id, representative_item_id, preview_path,
                  latest_captured_at, average_confidence, revision, is_partial, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                cluster_rows,
            )

            member_rows: list[tuple[str, str, str]] = []
            for cluster in clusters:
                cluster_id = str(cluster["id"])
                for item_id, detection_id in cluster.get("member_ids", []):
                    member_rows.append((cluster_id, str(item_id), str(detection_id)))
            if member_rows:
                conn.executemany(
                    """
                    INSERT INTO unknown_cluster_members (cluster_id, item_id, detection_id)
                    VALUES (?, ?, ?)
                    """,
                    member_rows,
                )
            self._delete_empty_unknown_clusters(conn)

    def mark_detections_cluster_clean(
        self,
        detections: Sequence[tuple[str, str]],
        *,
        cleaned_revision: str,
    ) -> None:
        buffered = [(item_id, detection_id) for item_id, detection_id in detections if item_id and detection_id]
        if not buffered:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                UPDATE detections
                SET cluster_dirty = 0,
                    cluster_dirty_revision = ?,
                    updated_at = ?
                WHERE item_id = ? AND detection_id = ?
                """,
                [
                    (cleaned_revision, cleaned_revision, item_id, detection_id)
                    for item_id, detection_id in buffered
                ],
            )

    def load_cached_unknown_clusters(
        self,
        kind: str,
        *,
        revision: str,
        partial: bool = False,
    ) -> list[dict[str, Any]] | None:
        cache_key = self._unknown_clusters_cache_key(kind, partial=partial)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM cache_entries WHERE cache_key = ? AND revision = ?",
                (cache_key, revision),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        return None

    def load_latest_cached_unknown_clusters(
        self,
        kind: str,
        *,
        partial: bool = False,
    ) -> list[dict[str, Any]] | None:
        cache_key = self._unknown_clusters_cache_key(kind, partial=partial)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM cache_entries WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        return None

    def save_cached_unknown_clusters(
        self,
        kind: str,
        *,
        revision: str,
        clusters: list[dict[str, Any]],
        partial: bool = False,
    ) -> None:
        cache_key = self._unknown_clusters_cache_key(kind, partial=partial)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache_entries (cache_key, revision, payload, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                  revision=excluded.revision,
                  payload=excluded.payload,
                  updated_at=excluded.updated_at
                """,
                (
                    cache_key,
                    revision,
                    json.dumps(clusters, separators=(",", ":"), sort_keys=True),
                    revision,
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS items (
                  id TEXT PRIMARY KEY,
                  title TEXT NOT NULL,
                  media_kind TEXT NOT NULL,
                  captured_at TEXT NOT NULL,
                  modified_ts REAL NOT NULL,
                  favorite INTEGER NOT NULL,
                  hidden INTEGER NOT NULL,
                  path TEXT NOT NULL,
                  payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_items_captured_at ON items(captured_at DESC, modified_ts DESC);
                CREATE INDEX IF NOT EXISTS idx_items_kind ON items(media_kind);

                CREATE TABLE IF NOT EXISTS personas (
                  id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_personas_kind_name ON personas(kind, name);

                CREATE TABLE IF NOT EXISTS albums (
                  id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                  id TEXT PRIMARY KEY,
                  title TEXT NOT NULL,
                  memory_type TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  start_date TEXT NOT NULL,
                  end_date TEXT NOT NULL,
                  payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_dates ON memories(end_date DESC, created_at DESC);

                CREATE TABLE IF NOT EXISTS item_personas (
                  item_id TEXT NOT NULL,
                  persona_id TEXT NOT NULL,
                  PRIMARY KEY (item_id, persona_id)
                );
                CREATE INDEX IF NOT EXISTS idx_item_personas_persona ON item_personas(persona_id, item_id);

                CREATE TABLE IF NOT EXISTS item_tags (
                  item_id TEXT NOT NULL,
                  tag TEXT NOT NULL,
                  PRIMARY KEY (item_id, tag)
                );
                CREATE INDEX IF NOT EXISTS idx_item_tags_tag ON item_tags(tag, item_id);

                CREATE TABLE IF NOT EXISTS item_search (
                  item_id TEXT PRIMARY KEY,
                  search_blob TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS detections (
                  item_id TEXT NOT NULL,
                  detection_id TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  label TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  bbox_x INTEGER NOT NULL,
                  bbox_y INTEGER NOT NULL,
                  bbox_w INTEGER NOT NULL,
                  bbox_h INTEGER NOT NULL,
                  persona_id TEXT,
                  encoding_json TEXT NOT NULL,
                  signature TEXT NOT NULL,
                  source_frame_token TEXT NOT NULL,
                  captured_at TEXT NOT NULL,
                  pipeline_revision TEXT NOT NULL,
                  cluster_dirty INTEGER NOT NULL,
                  cluster_dirty_revision TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (item_id, detection_id),
                  FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_detections_kind_dirty
                  ON detections(kind, cluster_dirty, captured_at DESC);
                CREATE INDEX IF NOT EXISTS idx_detections_persona
                  ON detections(persona_id, kind, captured_at DESC);

                CREATE TABLE IF NOT EXISTS unknown_clusters (
                  cluster_id TEXT PRIMARY KEY,
                  kind TEXT NOT NULL,
                  label TEXT NOT NULL,
                  member_count INTEGER NOT NULL,
                  item_count INTEGER NOT NULL,
                  representative_detection_id TEXT NOT NULL,
                  representative_item_id TEXT NOT NULL,
                  preview_path TEXT NOT NULL,
                  latest_captured_at TEXT NOT NULL,
                  average_confidence REAL NOT NULL,
                  revision TEXT NOT NULL,
                  is_partial INTEGER NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_unknown_clusters_kind
                  ON unknown_clusters(kind, is_partial, member_count DESC, latest_captured_at DESC);

                CREATE TABLE IF NOT EXISTS unknown_cluster_members (
                  cluster_id TEXT NOT NULL,
                  item_id TEXT NOT NULL,
                  detection_id TEXT NOT NULL,
                  PRIMARY KEY (cluster_id, item_id, detection_id),
                  FOREIGN KEY (cluster_id) REFERENCES unknown_clusters(cluster_id) ON DELETE CASCADE,
                  FOREIGN KEY (item_id, detection_id) REFERENCES detections(item_id, detection_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_unknown_cluster_members_detection
                  ON unknown_cluster_members(item_id, detection_id);

                CREATE TABLE IF NOT EXISTS cache_entries (
                  cache_key TEXT PRIMARY KEY,
                  revision TEXT NOT NULL,
                  payload TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                """
            )

    def _migrate_legacy_json_if_needed(self) -> Path | None:
        if self._has_state():
            return None
        for candidate in self.legacy_json_candidates():
            legacy_state = JsonLibraryStore(candidate).load()
            if (
                not legacy_state.items
                and not legacy_state.personas
                and not legacy_state.albums
                and not legacy_state.memories
            ):
                continue
            self.save(legacy_state)
            return candidate
        return None

    def _has_state(self) -> bool:
        with self._connect() as conn:
            for table in ("items", "personas", "albums", "memories"):
                row = conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
                if row is not None:
                    return True
        return False

    def _metadata_value(self, conn: sqlite3.Connection, key: str) -> str | None:
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _set_metadata(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            """
            INSERT INTO metadata (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )

    def _replace_entity_table(
        self,
        conn: sqlite3.Connection,
        table: str,
        rows: Iterable[tuple[Any, ...]],
    ) -> None:
        buffered_rows = list(rows)
        conn.execute(f"DELETE FROM {table}")
        if not buffered_rows:
            return
        if table == "items":
            conn.executemany(
                """
                INSERT INTO items (
                  id, title, media_kind, captured_at, modified_ts, favorite, hidden, path, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                buffered_rows,
            )
            return
        if table == "personas":
            conn.executemany(
                "INSERT INTO personas (id, name, kind, created_at, payload) VALUES (?, ?, ?, ?, ?)",
                buffered_rows,
            )
            return
        if table == "albums":
            conn.executemany(
                "INSERT INTO albums (id, name, created_at, payload) VALUES (?, ?, ?, ?)",
                buffered_rows,
            )
            return
        if table == "memories":
            conn.executemany(
                """
                INSERT INTO memories (
                  id, title, memory_type, created_at, start_date, end_date, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                buffered_rows,
            )

    def _rebuild_item_indexes(
        self,
        conn: sqlite3.Connection,
        items: Iterable[MediaItem],
        personas: dict[str, Persona],
    ) -> None:
        conn.execute("DELETE FROM item_personas")
        conn.execute("DELETE FROM item_tags")
        conn.execute("DELETE FROM item_search")
        self._insert_item_indexes(conn, list(items), personas)

    def _rebuild_detection_rows(
        self,
        conn: sqlite3.Connection,
        items: Iterable[MediaItem],
        *,
        updated_at: str,
    ) -> None:
        buffered_items = list(items)
        conn.execute("DELETE FROM detections")
        if not buffered_items:
            self._touch_cluster_revision(conn, updated_at)
            return
        detection_rows: list[tuple[Any, ...]] = []
        for item in buffered_items:
            detection_rows.extend(self._detection_rows_for_item(item, updated_at=updated_at))
        if detection_rows:
            conn.executemany(
                """
                INSERT INTO detections (
                  item_id, detection_id, kind, label, confidence,
                  bbox_x, bbox_y, bbox_w, bbox_h, persona_id, encoding_json,
                  signature, source_frame_token, captured_at, pipeline_revision,
                  cluster_dirty, cluster_dirty_revision, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                detection_rows,
            )
        self._touch_cluster_revision(conn, updated_at)

    def _replace_item_detection_rows(
        self,
        conn: sqlite3.Connection,
        items: Sequence[MediaItem],
        *,
        updated_at: str,
    ) -> None:
        item_ids = [item.id for item in items if item.id]
        if item_ids:
            placeholders = ",".join("?" for _ in item_ids)
            conn.execute(f"DELETE FROM detections WHERE item_id IN ({placeholders})", item_ids)
        detection_rows: list[tuple[Any, ...]] = []
        for item in items:
            detection_rows.extend(self._detection_rows_for_item(item, updated_at=updated_at))
        if detection_rows:
            conn.executemany(
                """
                INSERT INTO detections (
                  item_id, detection_id, kind, label, confidence,
                  bbox_x, bbox_y, bbox_w, bbox_h, persona_id, encoding_json,
                  signature, source_frame_token, captured_at, pipeline_revision,
                  cluster_dirty, cluster_dirty_revision, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                detection_rows,
            )
        self._touch_cluster_revision(conn, updated_at)

    def _insert_item_indexes(
        self,
        conn: sqlite3.Connection,
        items: Sequence[MediaItem],
        personas: dict[str, Persona],
    ) -> None:
        item_persona_rows: list[tuple[str, str]] = []
        item_tag_rows: list[tuple[str, str]] = []
        item_search_rows: list[tuple[str, str]] = []
        for item in items:
            persona_ids = self._item_persona_ids(item)
            item_persona_rows.extend((item.id, persona_id) for persona_id in persona_ids)
            item_tag_rows.extend((item.id, tag.lower()) for tag in item.tags if tag)
            item_search_rows.append((item.id, self._search_blob(item, personas, persona_ids)))
        if item_persona_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO item_personas (item_id, persona_id) VALUES (?, ?)",
                item_persona_rows,
            )
        if item_tag_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO item_tags (item_id, tag) VALUES (?, ?)",
                item_tag_rows,
            )
        if item_search_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO item_search (item_id, search_blob) VALUES (?, ?)",
                item_search_rows,
            )

    def _delete_item_indexes(self, conn: sqlite3.Connection, item_ids: Sequence[str]) -> None:
        buffered_ids = [item_id for item_id in item_ids if item_id]
        if not buffered_ids:
            return
        placeholders = ",".join("?" for _ in buffered_ids)
        conn.execute(f"DELETE FROM item_personas WHERE item_id IN ({placeholders})", buffered_ids)
        conn.execute(f"DELETE FROM item_tags WHERE item_id IN ({placeholders})", buffered_ids)
        conn.execute(f"DELETE FROM item_search WHERE item_id IN ({placeholders})", buffered_ids)

    def _clear_cache_entries(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM cache_entries")

    def _delete_empty_unknown_clusters(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            DELETE FROM unknown_clusters
            WHERE cluster_id NOT IN (
              SELECT DISTINCT cluster_id FROM unknown_cluster_members
            )
            """
        )

    def _item_row(self, item: MediaItem) -> tuple[Any, ...]:
        return (
            item.id,
            item.title,
            item.media_kind,
            item.captured_at,
            item.modified_ts,
            int(item.favorite),
            int(item.hidden),
            item.path,
            json.dumps(item.to_dict(), separators=(",", ":"), sort_keys=True),
        )

    def _persona_row(self, persona: Persona) -> tuple[Any, ...]:
        return (
            persona.id,
            persona.name,
            persona.kind,
            persona.created_at,
            json.dumps(persona.to_dict(), separators=(",", ":"), sort_keys=True),
        )

    def _album_row(self, album: Album) -> tuple[Any, ...]:
        return (
            album.id,
            album.name,
            album.created_at,
            json.dumps(album.to_dict(), separators=(",", ":"), sort_keys=True),
        )

    def _memory_row(self, memory: Memory) -> tuple[Any, ...]:
        return (
            memory.id,
            memory.title,
            memory.memory_type,
            memory.created_at,
            memory.start_date,
            memory.end_date,
            json.dumps(memory.to_dict(), separators=(",", ":"), sort_keys=True),
        )

    def _item_persona_ids(self, item: MediaItem) -> list[str]:
        persona_ids = set(item.manual_persona_ids)
        for detection in item.detections:
            if detection.persona_id:
                persona_ids.add(detection.persona_id)
        return sorted(persona_ids)

    def _search_blob(
        self,
        item: MediaItem,
        personas: dict[str, Persona],
        persona_ids: Sequence[str],
    ) -> str:
        persona_names = " ".join(
            personas[persona_id].name
            for persona_id in persona_ids
            if persona_id in personas
        )
        detections = " ".join(region.label for region in item.detections)
        metadata_text = " ".join(f"{key} {value}" for key, value in item.metadata.items())
        parts = [
            item.title,
            item.relative_key,
            item.media_kind,
            " ".join(item.tags),
            persona_names,
            detections,
            metadata_text,
            item.notes,
        ]
        return " ".join(part.lower() for part in parts if part)

    def _detection_rows_for_item(
        self,
        item: MediaItem,
        *,
        updated_at: str,
    ) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        for detection in item.detections:
            bbox = list(detection.bbox) + [0, 0, 0, 0]
            if detection.kind == "face":
                pipeline_revision = str(item.metadata.get("human_face_pipeline", ""))
            elif detection.kind.startswith("pet"):
                pipeline_revision = "|".join(
                    value
                    for value in (
                        str(item.metadata.get("pet_face_model", "")),
                        str(item.metadata.get("pet_embedding_model", "")),
                    )
                    if value
                )
            else:
                pipeline_revision = str(item.metadata.get("object_model", ""))
            rows.append(
                (
                    item.id,
                    detection.id,
                    detection.kind,
                    detection.label,
                    float(detection.confidence),
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                    detection.persona_id,
                    json.dumps([float(value) for value in detection.encoding], separators=(",", ":")),
                    detection.signature or "",
                    "",
                    item.captured_at,
                    pipeline_revision,
                    1,
                    updated_at,
                    updated_at,
                )
            )
        return rows

    def _search_tokens(self, search_text: str) -> list[str]:
        return [token.lower() for token in search_text.split() if token.strip()]

    def _touch_cluster_revision(self, conn: sqlite3.Connection, revision: str) -> None:
        self._set_metadata(conn, "unknown_clusters_dirty_revision", revision)

    def _backfill_detections_if_needed(self) -> None:
        with self._connect() as conn:
            item_count_row = conn.execute("SELECT COUNT(*) AS count FROM items").fetchone()
            detection_count_row = conn.execute("SELECT COUNT(*) AS count FROM detections").fetchone()
            item_count = int(item_count_row["count"]) if item_count_row is not None else 0
            detection_count = int(detection_count_row["count"]) if detection_count_row is not None else 0
            if item_count == 0 or detection_count > 0:
                return
            updated_at = self._metadata_value(conn, "updated_at") or LibraryState().updated_at
            rows = conn.execute("SELECT payload FROM items").fetchall()
            detection_rows: list[tuple[Any, ...]] = []
            for row in rows:
                item = MediaItem.from_dict(json.loads(row["payload"]))
                detection_rows.extend(self._detection_rows_for_item(item, updated_at=updated_at))
            if detection_rows:
                conn.executemany(
                    """
                    INSERT INTO detections (
                      item_id, detection_id, kind, label, confidence,
                      bbox_x, bbox_y, bbox_w, bbox_h, persona_id, encoding_json,
                      signature, source_frame_token, captured_at, pipeline_revision,
                      cluster_dirty, cluster_dirty_revision, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    detection_rows,
                )
            self._touch_cluster_revision(conn, updated_at)

    def _fetch_payload_row(self, table: str, entity_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                f"SELECT payload FROM {table} WHERE id = ?",
                (entity_id,),
            ).fetchone()

    def _row_to_detection_record(self, row: sqlite3.Row) -> DetectionRecord:
        updated_at = row["updated_at"]
        if "detection_updated_at" in row.keys():
            updated_at = row["detection_updated_at"]
        kind = row["kind"]
        if "detection_kind" in row.keys():
            kind = row["detection_kind"]
        label = row["label"]
        if "detection_label" in row.keys():
            label = row["detection_label"]
        return DetectionRecord(
            item_id=str(row["item_id"]),
            detection_id=str(row["detection_id"]),
            kind=str(kind),
            label=str(label),
            confidence=float(row["confidence"]),
            bbox=[
                int(row["bbox_x"]),
                int(row["bbox_y"]),
                int(row["bbox_w"]),
                int(row["bbox_h"]),
            ],
            persona_id=(None if row["persona_id"] in {None, ""} else str(row["persona_id"])),
            encoding=[
                float(value)
                for value in json.loads(str(row["encoding_json"] or "[]"))
            ],
            signature=(None if not row["signature"] else str(row["signature"])),
            source_frame_token=str(row["source_frame_token"] or ""),
            captured_at=str(row["captured_at"] or ""),
            pipeline_revision=str(row["pipeline_revision"] or ""),
            cluster_dirty=bool(int(row["cluster_dirty"])),
            cluster_dirty_revision=str(row["cluster_dirty_revision"] or ""),
            updated_at=str(updated_at or ""),
        )

    def _unknown_clusters_cache_key(self, kind: str, *, partial: bool = False) -> str:
        suffix = "partial" if partial else "final"
        return f"unknown_clusters:{kind}:{suffix}"

    def _sqlite_path_for(self, path: Path) -> Path:
        if path.suffix.lower() in SQLITE_SUFFIXES:
            return path
        return path.with_suffix(".sqlite3")

    def _legacy_json_paths_for(self, path: Path) -> list[Path]:
        candidates: list[Path] = []
        if path.suffix.lower() == ".json":
            candidates.append(path)
        else:
            candidates.append(path.with_suffix(".json"))
        candidates.append(path.parent / "library.json")

        unique_candidates: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        return unique_candidates
