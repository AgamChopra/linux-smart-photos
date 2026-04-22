from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

from ..config import AppConfig

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


@dataclass(slots=True)
class ModelSpec:
    id: str
    role: str
    title: str
    source_url: str
    download_url: str
    storage_kind: str
    local_path: str
    summary: str
    license_note: str
    recommended: bool = False
    repo_id: str = ""


@dataclass(slots=True)
class ModelStatus:
    id: str
    role: str
    title: str
    installed: bool
    local_path: str
    source_url: str
    download_url: str
    summary: str
    license_note: str


class ModelManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.models_path.mkdir(parents=True, exist_ok=True)

    def catalog(self) -> list[ModelSpec]:
        return [
            ModelSpec(
                id="ultralytics_yolo11n",
                role="Object Detection",
                title="Ultralytics YOLO11n",
                source_url="https://docs.ultralytics.com/models/yolo11/",
                download_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt",
                storage_kind="file",
                local_path="ultralytics/yolo11n.pt",
                summary="Lightweight default detector for scenes, objects, pets, and search tags.",
                license_note="Ultralytics models are distributed under AGPL-3.0 or Enterprise terms.",
                recommended=True,
            ),
            ModelSpec(
                id="insightface_buffalo_sc",
                role="Human Personas",
                title="InsightFace buffalo_sc",
                source_url="https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md",
                download_url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
                storage_kind="zip_dir",
                local_path="insightface/models/buffalo_sc",
                summary="Fast and small human face detection and recognition pack for persona creation.",
                license_note="InsightFace pretrained model packs are for non-commercial research purposes only.",
                recommended=True,
            ),
            ModelSpec(
                id="avito_dinov2_small_petface",
                role="Pet Personas",
                title="AvitoTech DINOv2-Small Animal ID",
                source_url="https://huggingface.co/AvitoTech/DINO-v2-small-for-animal-identification",
                download_url="https://huggingface.co/AvitoTech/DINO-v2-small-for-animal-identification/resolve/main/model.safetensors",
                storage_kind="huggingface_snapshot",
                local_path="huggingface/AvitoTech__DINO-v2-small-for-animal-identification",
                summary="Fine-tuned animal-identification embeddings for cat and dog persona matching.",
                license_note="Review the Hugging Face model card and upstream licenses before use.",
                recommended=True,
                repo_id="AvitoTech/DINO-v2-small-for-animal-identification",
            ),
            ModelSpec(
                id="lostpet_yolov7_pet_face",
                role="Pet Face Detector",
                title="LostPetInitiative YOLOv7 Pet Face Detector",
                source_url="https://zenodo.org/records/7607110",
                download_url="https://zenodo.org/records/7607110/files/yolov7-pet-face.pt?download=1",
                storage_kind="file",
                local_path="pet-detectors/yolov7-pet-face.pt",
                summary="Dedicated cat-first pet face detector checkpoint for cat and dog persona crops.",
                license_note="Zenodo record lists CC BY 4.0; verify downstream compatibility before redistribution.",
                recommended=True,
            ),
        ]

    def recommended_specs(self) -> list[ModelSpec]:
        return [spec for spec in self.catalog() if spec.recommended]

    def get_spec(self, model_id: str) -> ModelSpec:
        for spec in self.catalog():
            if spec.id == model_id:
                return spec
        raise KeyError(model_id)

    def resolved_path(self, spec: ModelSpec) -> Path:
        return self.config.models_path / spec.local_path

    def status(self, model_id: str) -> ModelStatus:
        spec = self.get_spec(model_id)
        local_path = self.resolved_path(spec)
        installed = local_path.exists()
        return ModelStatus(
            id=spec.id,
            role=spec.role,
            title=spec.title,
            installed=installed,
            local_path=str(local_path),
            source_url=spec.source_url,
            download_url=spec.download_url,
            summary=spec.summary,
            license_note=spec.license_note,
        )

    def all_statuses(self) -> list[ModelStatus]:
        return [self.status(spec.id) for spec in self.catalog()]

    def ensure_model(self, model_id: str) -> str:
        spec = self.get_spec(model_id)
        local_path = self.resolved_path(spec)
        if local_path.exists():
            return str(local_path)
        if not self.config.auto_download_models:
            return ""
        return self.download_model(model_id)

    def download_recommended_models(self) -> list[str]:
        paths: list[str] = []
        for spec in self.recommended_specs():
            paths.append(self.download_model(spec.id))
        return paths

    def download_model(self, model_id: str) -> str:
        spec = self.get_spec(model_id)
        local_path = self.resolved_path(spec)
        if local_path.exists():
            return str(local_path)

        if spec.storage_kind == "huggingface_snapshot":
            if snapshot_download is None:
                raise RuntimeError(
                    "huggingface-hub is not installed. Install the AI extras to download this model."
                )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=spec.repo_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
            )
            return str(local_path)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="linux-smart-photos-model-") as temp_dir:
            temp_path = Path(temp_dir) / Path(spec.download_url.split("?")[0]).name
            self._download_file(spec.download_url, temp_path)
            if spec.storage_kind == "zip_dir":
                extract_root = local_path.parent
                extract_root.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(temp_path) as archive:
                    archive.extractall(extract_root)
                return str(local_path)
            shutil.move(str(temp_path), str(local_path))
        return str(local_path)

    def _download_file(self, url: str, destination: Path) -> None:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "linux-smart-photos/0.1"},
        )
        with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
