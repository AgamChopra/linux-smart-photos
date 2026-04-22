from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".avif",
}
VIDEO_EXTENSIONS = {
    ".mov",
    ".mp4",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
}
GIF_EXTENSIONS = {".gif"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | GIF_EXTENSIONS


@dataclass(slots=True)
class MediaAssetSpec:
    id: str
    relative_key: str
    title: str
    media_kind: str
    extension: str
    display_path: str
    component_paths: list[str]
    file_signature: str
    size_bytes: int
    modified_ts: float


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def media_kind_for_path(path: Path) -> str | None:
    extension = path.suffix.lower()
    if extension in GIF_EXTENSIONS:
        return "gif"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    if extension in IMAGE_EXTENSIONS:
        return "image"
    return None


def build_signature(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha1()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def build_asset_specs(root: Path) -> dict[str, MediaAssetSpec]:
    root = root.expanduser().resolve()
    if not root.exists():
        return {}

    files = sorted(path for path in root.rglob("*") if path.is_file() and is_supported(path))
    grouped: dict[tuple[Path, str], list[Path]] = {}
    for path in files:
        grouped.setdefault((path.parent, path.stem), []).append(path)

    used: set[Path] = set()
    assets: dict[str, MediaAssetSpec] = {}

    for (_, stem), candidates in grouped.items():
        image_candidates = [
            path for path in candidates if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        video_candidates = [
            path for path in candidates if path.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not image_candidates or not video_candidates:
            continue

        preview = sorted(image_candidates)[0]
        rel_key = str(preview.relative_to(root).with_suffix(""))
        item_id = stable_id(rel_key)
        component_paths = sorted(candidates)
        size_bytes = sum(path.stat().st_size for path in component_paths)
        modified_ts = max(path.stat().st_mtime for path in component_paths)
        assets[item_id] = MediaAssetSpec(
            id=item_id,
            relative_key=rel_key,
            title=preview.stem.replace("_", " ").replace("-", " ").strip() or preview.stem,
            media_kind="live_photo",
            extension=preview.suffix.lower(),
            display_path=str(preview),
            component_paths=[str(path) for path in component_paths],
            file_signature=build_signature(component_paths, root),
            size_bytes=size_bytes,
            modified_ts=modified_ts,
        )
        used.update(component_paths)

    for path in files:
        if path in used:
            continue
        kind = media_kind_for_path(path)
        if not kind:
            continue
        rel_key = str(path.relative_to(root))
        item_id = stable_id(rel_key)
        assets[item_id] = MediaAssetSpec(
            id=item_id,
            relative_key=rel_key,
            title=path.stem.replace("_", " ").replace("-", " ").strip() or path.stem,
            media_kind=kind,
            extension=path.suffix.lower(),
            display_path=str(path),
            component_paths=[str(path)],
            file_signature=build_signature([path], root),
            size_bytes=path.stat().st_size,
            modified_ts=path.stat().st_mtime,
        )

    return assets
