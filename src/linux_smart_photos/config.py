from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


APP_DIR_NAME = "linux-smart-photos"


@dataclass(slots=True)
class AppConfig:
    media_root: str
    database_path: str
    cache_dir: str
    models_dir: str
    thumbnail_size: int = 256
    face_recognition_enabled: bool = True
    object_detection_enabled: bool = True
    pet_recognition_enabled: bool = True
    auto_generate_memories: bool = True
    auto_download_models: bool = False
    video_ai_enabled: bool = True
    video_frame_sample_seconds: float = 2.0
    video_max_analysis_frames: int = 12
    scan_batch_size: int = 8
    analysis_batch_size: int = 24
    pet_embedding_batch_size: int = 64
    prefetch_workers: int = 4
    gallery_thumbnail_cache_mb: int = 2048
    gallery_prefetch_all_thumbnails: bool = True
    gallery_prefetch_page_delay_ms: int = 75
    face_match_threshold: float = 0.42
    face_embedding_similarity_threshold: float = 0.57
    pet_embedding_similarity_threshold: float = 0.70
    pet_hash_distance_threshold: int = 14
    memory_min_items: int = 6
    object_model_path: str = ""
    object_model_id: str = "ultralytics_yolo11n"
    human_face_detector_model_id: str = "insightface_antelope"
    human_face_detector_path: str = ""
    human_face_detector_backend: str = "auto"
    human_face_model_id: str = "insightface_buffalo_sc"
    pet_detector_model_id: str = "lostpet_yolov7_pet_face"
    pet_embedding_model_id: str = "avito_dinov2_small_petface"

    @property
    def media_root_path(self) -> Path:
        return Path(self.media_root).expanduser().resolve()

    @property
    def database_file(self) -> Path:
        return Path(self.database_path).expanduser().resolve()

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir).expanduser().resolve()

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir).expanduser().resolve()


def default_paths() -> tuple[Path, Path, Path]:
    config_dir = Path.home() / ".config" / APP_DIR_NAME
    data_dir = Path.home() / ".local" / "share" / APP_DIR_NAME
    cache_dir = Path.home() / ".cache" / APP_DIR_NAME
    return config_dir, data_dir, cache_dir


def default_media_root() -> Path:
    cwd_root = (Path.cwd() / "file" / "photos").resolve()
    if cwd_root.exists():
        return cwd_root
    return (Path.home() / "file" / "photos").resolve()


def config_file_path() -> Path:
    config_dir, _, _ = default_paths()
    return config_dir / "config.json"


def default_config() -> AppConfig:
    _, data_dir, cache_dir = default_paths()
    return AppConfig(
        media_root=str(default_media_root()),
        database_path=str(data_dir / "library.sqlite3"),
        cache_dir=str(cache_dir / "thumbnails"),
        models_dir=str(data_dir / "models"),
    )


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or config_file_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        config = default_config()
        write_config(config, config_path)
        return config

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    merged_payload = asdict(default_config())
    merged_payload.update(payload)
    config = AppConfig(**merged_payload)
    config.database_file.parent.mkdir(parents=True, exist_ok=True)
    config.cache_path.mkdir(parents=True, exist_ok=True)
    config.models_path.mkdir(parents=True, exist_ok=True)
    return config


def normalize_config_file(path: Path | None = None) -> tuple[AppConfig, bool]:
    config_path = path or config_file_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        config = default_config()
        write_config(config, config_path)
        return config, True

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    merged_payload = asdict(default_config())
    merged_payload.update(payload)
    config = AppConfig(**merged_payload)
    normalized = payload != merged_payload
    if normalized:
        write_config(config, config_path)
    else:
        config.database_file.parent.mkdir(parents=True, exist_ok=True)
        config.cache_path.mkdir(parents=True, exist_ok=True)
        config.models_path.mkdir(parents=True, exist_ok=True)
    return config, normalized


def write_config(config: AppConfig, path: Path | None = None) -> None:
    config_path = path or config_file_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.database_file.parent.mkdir(parents=True, exist_ok=True)
    config.cache_path.mkdir(parents=True, exist_ok=True)
    config.models_path.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
