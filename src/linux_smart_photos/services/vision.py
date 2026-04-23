from __future__ import annotations

from dataclasses import dataclass, field
import ctypes
import ctypes.util
import logging
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None

import numpy as np
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

try:
    import imagehash
except Exception:
    imagehash = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import onnx
except Exception:
    onnx = None

from ..config import AppConfig
from ..media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, MediaAssetSpec
from ..models import DetectionRegion
from .human_face_backend import HUMAN_FACE_PIPELINE_VERSION, HumanFaceBackend
from .model_manager import ModelManager

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

logging.getLogger("ultralytics").setLevel(logging.ERROR)

try:
    import torch
    from transformers import AutoImageProcessor, AutoModel
except Exception:
    torch = None
    AutoImageProcessor = None
    AutoModel = None


CAT_LABELS = {"cat", "kitten"}
DOG_LABELS = {"dog", "puppy"}
PET_LABELS = CAT_LABELS | DOG_LABELS | {"bird", "horse", "rabbit", "hamster", "guinea pig", "ferret"}
ANALYSIS_IMAGE_MAX_DIMENSION = 1600
VIDEO_ANALYSIS_FRAME_MAX_DIMENSION = 1280


@dataclass(slots=True)
class AnalysisResult:
    tags: list[str]
    detections: list[DetectionRegion]
    metadata: dict[str, Any]


@dataclass(slots=True)
class VideoFrameSample:
    sample_number: int
    frame_index: int
    timestamp_seconds: float
    image: Image.Image


@dataclass(slots=True)
class PreparedAssetInput:
    spec: MediaAssetSpec
    still_image: Image.Image | None = None
    video_frames: list[VideoFrameSample] = field(default_factory=list)
    video_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PetFaceCandidate:
    label: str
    confidence: float
    bbox: list[int]


@dataclass(slots=True)
class _AnalysisUnit:
    asset_index: int
    image: Image.Image
    unit_kind: str
    video_prefix: str = ""


class PetEmbeddingModel:
    def __init__(self, model_dir: Path) -> None:
        if torch is None or AutoModel is None or AutoImageProcessor is None:
            raise RuntimeError("Pet embedding dependencies are not installed.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device == "cuda"
        self.processor = AutoImageProcessor.from_pretrained(str(model_dir), use_fast=True)
        self.model = AutoModel.from_pretrained(str(model_dir)).to(self.device).eval()
        if self.use_half:
            self.model = self.model.half()

    def embed(self, image: Image.Image) -> list[float]:
        embeddings = self.embed_batch([image])
        return embeddings[0] if embeddings else []

    def embed_batch(self, images: list[Image.Image]) -> list[list[float]]:
        if not images:
            return []

        inputs = self.processor(images=images, return_tensors="pt")
        converted_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            tensor = value.to(self.device)
            if self.use_half and torch.is_floating_point(tensor):
                tensor = tensor.half()
            converted_inputs[key] = tensor
        with torch.no_grad():
            outputs = self.model(**converted_inputs)

        if hasattr(outputs, "last_hidden_state"):
            tensor = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output"):
            tensor = outputs.pooler_output
        else:
            raise RuntimeError("Pet embedding model returned an unsupported output structure.")

        tensor = torch.nn.functional.normalize(tensor, dim=1)
        vectors = tensor.detach().cpu().numpy()
        return [
            [round(float(value), 6) for value in vector.tolist()]
            for vector in vectors
        ]


class VisionAnalyzer:
    def __init__(self, config: AppConfig, model_manager: ModelManager | None = None) -> None:
        self.config = config
        self.model_manager = model_manager or ModelManager(config)
        self.human_face_providers: list[str] = []
        self.human_face_uses_gpu = False
        self.human_face_device_label = ""
        self.human_face_detector_name = ""
        self.human_face_recognizer_name = ""
        self.human_face_pipeline_id = self.human_face_pipeline_revision(config)
        self.yolo_device = self._preferred_yolo_device()
        self.cat_face_detector = self._load_cat_face_detector()
        self.human_face_backend = self._load_human_face_backend()
        self.object_model = self._load_object_model()
        self.pet_face_model = self._load_pet_face_model()
        self.pet_embedding_model = self._load_pet_embedding_model()

    def analyze(self, spec: MediaAssetSpec, *, analysis_mode: str = "full") -> AnalysisResult:
        if spec.media_kind == "video":
            video_result = self._analyze_video(spec, analysis_mode=analysis_mode)
            if video_result.tags or video_result.detections or video_result.metadata.get("video_ai_frames_analyzed"):
                return video_result

        if spec.media_kind == "live_photo":
            still_image = self.load_preview_image(spec)
            still_result = (
                self._analyze_still_image(still_image, analysis_mode=analysis_mode)
                if still_image is not None
                else self._empty_analysis()
            )
            video_result = self._analyze_video(spec, analysis_mode=analysis_mode)
            if video_result.tags or video_result.detections or video_result.metadata.get("video_ai_frames_analyzed"):
                return self._merge_analysis_results(still_result, video_result)
            return still_result

        image = self.load_preview_image(spec)
        return self._analyze_still_image(image, analysis_mode=analysis_mode)

    def _empty_analysis(self) -> AnalysisResult:
        return AnalysisResult(tags=[], detections=[], metadata={})

    def _analyze_still_image(
        self,
        image: Image.Image | None,
        *,
        analysis_mode: str = "full",
    ) -> AnalysisResult:
        if image is None:
            return self._empty_analysis()

        tags: set[str] = set()
        detections: list[DetectionRegion] = []

        human_face_detections = self._detect_human_faces(image)
        if human_face_detections:
            tags.update({"face", "person"})
            detections.extend(human_face_detections)

        if analysis_mode == "human_faces_only":
            return AnalysisResult(
                tags=sorted(tags),
                detections=detections,
                metadata=self._analysis_metadata(image),
            )

        object_detections, object_tags = self._detect_objects(image)
        pet_detections, pet_tags = self._build_pet_detections(image, object_detections)
        non_pet_object_detections = [
            detection
            for detection in object_detections
            if detection.label.lower() not in PET_LABELS
        ]
        detections.extend(pet_detections)
        detections.extend(non_pet_object_detections)
        tags.update(object_tags - PET_LABELS)
        tags.update(pet_tags)

        return AnalysisResult(
            tags=sorted(tags),
            detections=detections,
            metadata=self._analysis_metadata(image),
        )

    def _merge_analysis_results(self, left: AnalysisResult, right: AnalysisResult) -> AnalysisResult:
        detections = list(left.detections)
        self._merge_unique_detections(detections, right.detections)
        return AnalysisResult(
            tags=sorted(set(left.tags) | set(right.tags)),
            detections=detections,
            metadata=left.metadata | right.metadata,
        )

    def _analysis_metadata(self, image: Image.Image) -> dict[str, Any]:
        return {
            "analyzed_width": image.size[0],
            "analyzed_height": image.size[1],
            "human_face_model": (
                f"{self.human_face_detector_name}+{self.human_face_recognizer_name}"
                if self.human_face_backend
                else ""
            ),
            "human_face_pipeline": self.human_face_pipeline_id if self.human_face_backend else "",
            "human_face_providers": list(self.human_face_providers),
            "human_face_device": self.human_face_device_label if self.human_face_backend else "",
            "human_face_detector_model": self.human_face_detector_name if self.human_face_backend else "",
            "human_face_recognizer_model": self.human_face_recognizer_name if self.human_face_backend else "",
            "object_model": self.config.object_model_id if self.object_model else "",
            "object_device": self._yolo_device_label() if self.object_model else "",
            "pet_face_model": self.config.pet_detector_model_id if self.pet_face_model else "",
            "pet_face_device": self._yolo_device_label() if self.pet_face_model else "",
            "pet_embedding_model": self.config.pet_embedding_model_id if self.pet_embedding_model else "",
            "pet_embedding_device": getattr(self.pet_embedding_model, "device", ""),
            "cat_face_fallback": bool(self.cat_face_detector),
        }

    def analyze_batch(
        self,
        assets: list[PreparedAssetInput],
        *,
        analysis_mode: str = "full",
    ) -> list[AnalysisResult]:
        if not assets:
            return []

        units: list[_AnalysisUnit] = []
        for asset_index, asset in enumerate(assets):
            if asset.spec.media_kind == "video":
                for frame in asset.video_frames:
                    units.append(
                        _AnalysisUnit(
                            asset_index=asset_index,
                            image=frame.image,
                            unit_kind="video",
                            video_prefix=f"video-{frame.sample_number:02d}-{frame.timestamp_seconds:07.2f}",
                        )
                    )
                continue

            if asset.still_image is not None:
                units.append(
                    _AnalysisUnit(
                        asset_index=asset_index,
                        image=asset.still_image,
                        unit_kind="still",
                    )
                )

            if asset.spec.media_kind == "live_photo":
                for frame in asset.video_frames:
                    units.append(
                        _AnalysisUnit(
                            asset_index=asset_index,
                            image=frame.image,
                            unit_kind="video",
                            video_prefix=f"video-{frame.sample_number:02d}-{frame.timestamp_seconds:07.2f}",
                        )
                    )

        unit_results = self._analyze_units_batch(units, analysis_mode=analysis_mode)
        grouped_units: list[list[tuple[_AnalysisUnit, AnalysisResult]]] = [[] for _ in assets]
        for unit, unit_result in zip(units, unit_results, strict=False):
            grouped_units[unit.asset_index].append((unit, unit_result))

        aggregated: list[AnalysisResult] = []
        for asset_index, asset in enumerate(assets):
            still_result = self._empty_analysis()
            video_tags: set[str] = set()
            video_detections: list[DetectionRegion] = []
            analyzed_video_image: Image.Image | None = None

            for unit, unit_result in grouped_units[asset_index]:
                if unit.unit_kind == "still":
                    still_result = unit_result
                    continue

                frame_detections = [
                    detection
                    for detection in unit_result.detections
                    if detection.kind == "face" or self._is_pet_detection(detection)
                ]
                video_tags.update(unit_result.tags)
                self._merge_unique_detections(
                    video_detections,
                    self._prefix_detections(frame_detections, prefix=unit.video_prefix),
                )
                if analyzed_video_image is None:
                    analyzed_video_image = unit.image

            if asset.spec.media_kind == "video":
                aggregated.append(
                    self._build_video_analysis_result(
                        analyzed_video_image=analyzed_video_image,
                        tags=video_tags,
                        detections=video_detections,
                        video_metadata=asset.video_metadata,
                    )
                )
                continue

            if asset.spec.media_kind == "live_photo":
                video_result = self._build_video_analysis_result(
                    analyzed_video_image=analyzed_video_image,
                    tags=video_tags,
                    detections=video_detections,
                    video_metadata=asset.video_metadata,
                )
                if video_result.tags or video_result.detections or video_result.metadata.get("video_ai_frames_analyzed"):
                    aggregated.append(self._merge_analysis_results(still_result, video_result))
                else:
                    aggregated.append(still_result)
                continue

            aggregated.append(still_result)

        return aggregated

    def load_video_analysis_frames(
        self,
        spec: MediaAssetSpec,
    ) -> tuple[list[VideoFrameSample], dict[str, Any]]:
        if cv2 is None or not self.config.video_ai_enabled:
            return [], {}

        video_path = self.primary_video_path(spec)
        if not video_path:
            return [], {}

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return [], {}

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            sample_indices = self._build_video_sample_indices(frame_count, fps)
            samples: list[VideoFrameSample] = []
            sampled_timestamps: list[float] = []

            for sample_number, frame_index in enumerate(sample_indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_seconds = frame_index / fps if fps > 0 else float(sample_number)
                frame_image = self._resize_for_analysis(
                    Image.fromarray(rgb_frame),
                    VIDEO_ANALYSIS_FRAME_MAX_DIMENSION,
                )
                samples.append(
                    VideoFrameSample(
                        sample_number=sample_number,
                        frame_index=frame_index,
                        timestamp_seconds=round(timestamp_seconds, 2),
                        image=frame_image,
                    )
                )
                sampled_timestamps.append(round(timestamp_seconds, 2))

            metadata = {
                "video_ai_frames_analyzed": len(sampled_timestamps),
                "video_ai_sample_timestamps": sampled_timestamps,
                "video_ai_mode": "interval_sampling",
                "video_ai_frame_count": frame_count,
                "video_ai_fps": round(fps, 3) if fps > 0 else 0.0,
            }
            return samples, metadata
        finally:
            capture.release()

    def _analyze_video(self, spec: MediaAssetSpec, *, analysis_mode: str = "full") -> AnalysisResult:
        video_frames, video_metadata = self.load_video_analysis_frames(spec)
        return self.analyze_batch(
            [
                PreparedAssetInput(
                    spec=spec,
                    video_frames=video_frames,
                    video_metadata=video_metadata,
                )
            ],
            analysis_mode=analysis_mode,
        )[0]

    def _build_video_sample_indices(self, frame_count: int, fps: float) -> list[int]:
        max_frames = max(1, int(self.config.video_max_analysis_frames))
        if frame_count <= max_frames:
            return list(range(frame_count))

        if fps > 0:
            sample_seconds = max(0.5, float(self.config.video_frame_sample_seconds))
            stride = max(1, int(round(fps * sample_seconds)))
            indices = list(range(0, frame_count, stride))
        else:
            stride = max(1, frame_count // max_frames)
            indices = list(range(0, frame_count, stride))

        if not indices or indices[-1] != frame_count - 1:
            indices.append(frame_count - 1)
        if indices[0] != 0:
            indices.insert(0, 0)
        if len(indices) > max_frames:
            selection = np.linspace(0, len(indices) - 1, num=max_frames, dtype=int)
            indices = [indices[index] for index in selection.tolist()]

        unique_indices = sorted({max(0, min(frame_count - 1, index)) for index in indices})
        return unique_indices[:max_frames]

    def _analyze_units_batch(
        self,
        units: list[_AnalysisUnit],
        *,
        analysis_mode: str = "full",
    ) -> list[AnalysisResult]:
        if not units:
            return []

        images = [unit.image for unit in units]
        human_face_batches = self._detect_human_faces_batch(images)
        object_batches = self._detect_objects_batch(images) if analysis_mode != "human_faces_only" else []
        pet_face_batches = self._detect_pet_face_candidates_batch(images) if analysis_mode != "human_faces_only" else []
        results: list[AnalysisResult] = []
        pet_embedding_requests: list[tuple[DetectionRegion, Image.Image]] = []

        for index, unit in enumerate(units):
            image = unit.image
            tags: set[str] = set()
            detections: list[DetectionRegion] = []

            human_face_detections = human_face_batches[index]
            if human_face_detections:
                tags.update({"face", "person"})
                detections.extend(human_face_detections)

            if analysis_mode == "human_faces_only":
                results.append(
                    AnalysisResult(
                        tags=sorted(tags),
                        detections=detections,
                        metadata=self._analysis_metadata(image),
                    )
                )
                continue

            object_detections, object_tags = object_batches[index]
            pet_detections, pet_tags = self._build_pet_detections_from_candidates(
                image,
                object_detections,
                pet_face_batches[index],
                embed=False,
            )
            for detection in pet_detections:
                crop = self.crop_region(image, detection.bbox)
                if crop is not None:
                    pet_embedding_requests.append((detection, crop))
            non_pet_object_detections = [
                detection
                for detection in object_detections
                if detection.label.lower() not in PET_LABELS
            ]
            detections.extend(pet_detections)
            detections.extend(non_pet_object_detections)
            tags.update(object_tags - PET_LABELS)
            tags.update(pet_tags)
            results.append(
                AnalysisResult(
                    tags=sorted(tags),
                    detections=detections,
                    metadata=self._analysis_metadata(image),
                )
            )

        if analysis_mode == "human_faces_only":
            return results

        embeddings = self._embed_pet_crops_batch([crop for _, crop in pet_embedding_requests])
        for (detection, _), embedding in zip(pet_embedding_requests, embeddings, strict=False):
            detection.encoding = embedding
        return results

    def _build_video_analysis_result(
        self,
        analyzed_video_image: Image.Image | None,
        tags: set[str],
        detections: list[DetectionRegion],
        video_metadata: dict[str, Any],
    ) -> AnalysisResult:
        if analyzed_video_image is None and not video_metadata.get("video_ai_frames_analyzed"):
            return self._empty_analysis()

        metadata = self._analysis_metadata(analyzed_video_image) if analyzed_video_image is not None else {}
        metadata.update(video_metadata)
        return AnalysisResult(
            tags=sorted(tags),
            detections=detections,
            metadata=metadata,
        )

    def _prefix_detections(
        self,
        detections: list[DetectionRegion],
        prefix: str,
    ) -> list[DetectionRegion]:
        return [
            DetectionRegion(
                id=f"{prefix}-{detection.id}",
                kind=detection.kind,
                label=detection.label,
                confidence=detection.confidence,
                bbox=list(detection.bbox),
                persona_id=detection.persona_id,
                encoding=list(detection.encoding),
                signature=detection.signature,
            )
            for detection in detections
        ]

    def _merge_unique_detections(
        self,
        existing: list[DetectionRegion],
        incoming: list[DetectionRegion],
    ) -> None:
        for candidate in incoming:
            match_index = next(
                (
                    index
                    for index, current in enumerate(existing)
                    if self._same_subject_detection(current, candidate)
                ),
                None,
            )
            if match_index is None:
                existing.append(candidate)
                continue

            current = existing[match_index]
            if self._detection_rank(candidate) > self._detection_rank(current):
                existing[match_index] = candidate
            elif current.label.lower() == "pet" and candidate.label.lower() in CAT_LABELS | DOG_LABELS:
                current.label = candidate.label
                if candidate.encoding and not current.encoding:
                    current.encoding = candidate.encoding
                if candidate.signature and not current.signature:
                    current.signature = candidate.signature

    def _same_subject_detection(self, left: DetectionRegion, right: DetectionRegion) -> bool:
        if left.kind == "face" and right.kind == "face":
            if left.encoding and right.encoding:
                return self._cosine_similarity(left.encoding, right.encoding) >= max(
                    self.config.face_embedding_similarity_threshold,
                    0.72,
                )
            if left.signature and right.signature:
                return self._signature_distance(left.signature, right.signature) <= 8
            return self._iou(left.bbox, right.bbox) >= 0.55

        if self._is_pet_detection(left) and self._is_pet_detection(right):
            if not self._pet_labels_compatible(left.label, right.label):
                return False
            if left.encoding and right.encoding:
                return self._cosine_similarity(left.encoding, right.encoding) >= max(
                    self.config.pet_embedding_similarity_threshold,
                    0.78,
                )
            if left.signature and right.signature:
                return self._signature_distance(left.signature, right.signature) <= 8
            return self._iou(left.bbox, right.bbox) >= 0.55

        if left.label.lower() != right.label.lower():
            return False
        return self._iou(left.bbox, right.bbox) >= 0.65

    def _detection_rank(self, detection: DetectionRegion) -> float:
        score = detection.confidence
        if detection.kind == "pet_face":
            score += 0.20
        elif detection.kind == "face":
            score += 0.15
        if detection.label.lower() in CAT_LABELS | DOG_LABELS:
            score += 0.08
        if detection.label.lower() != "pet":
            score += 0.04
        return score

    def _is_pet_detection(self, detection: DetectionRegion) -> bool:
        return detection.kind.startswith("pet") or detection.label.lower() in PET_LABELS | {"pet"}

    def _pet_labels_compatible(self, left: str, right: str) -> bool:
        normalized_left = left.lower()
        normalized_right = right.lower()
        if "pet" in {normalized_left, normalized_right}:
            return True
        if normalized_left in CAT_LABELS and normalized_right in CAT_LABELS:
            return True
        if normalized_left in DOG_LABELS and normalized_right in DOG_LABELS:
            return True
        return normalized_left == normalized_right

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return -1.0
        left_array = np.asarray(left, dtype=np.float32)
        right_array = np.asarray(right, dtype=np.float32)
        denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
        if denominator <= 0:
            return -1.0
        return float(np.dot(left_array, right_array) / denominator)

    def _signature_distance(self, left: str, right: str) -> int:
        if len(left) != len(right):
            return 64
        xor = int(left, 16) ^ int(right, 16)
        return xor.bit_count()

    def load_preview_image(self, spec: MediaAssetSpec) -> Image.Image | None:
        return self._load_image(spec)

    def load_analysis_image(self, spec: MediaAssetSpec) -> Image.Image | None:
        return self._load_image(spec, max_dimension=ANALYSIS_IMAGE_MAX_DIMENSION)

    def _load_image(
        self,
        spec: MediaAssetSpec,
        max_dimension: int | None = None,
    ) -> Image.Image | None:
        try:
            if spec.media_kind in {"image", "live_photo"}:
                image_path = self.primary_image_path(spec)
                if image_path:
                    with Image.open(image_path) as image:
                        loaded = ImageOps.exif_transpose(image).convert("RGB")
                        return self._resize_for_analysis(loaded, max_dimension)
                return None

            if spec.media_kind == "gif":
                with Image.open(spec.display_path) as image:
                    frame = next(ImageSequence.Iterator(image))
                    loaded = ImageOps.exif_transpose(frame).convert("RGB")
                    return self._resize_for_analysis(loaded, max_dimension)

            video_path = self.primary_video_path(spec)
            if not video_path or cv2 is None:
                return None

            capture = cv2.VideoCapture(str(video_path))
            try:
                ok, frame = capture.read()
                if not ok or frame is None:
                    return None
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                loaded = Image.fromarray(rgb_frame)
                return self._resize_for_analysis(loaded, max_dimension)
            finally:
                capture.release()
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            return None

    def _resize_for_analysis(
        self,
        image: Image.Image,
        max_dimension: int | None,
    ) -> Image.Image:
        if not max_dimension or max_dimension <= 0:
            return image
        width, height = image.size
        largest = max(width, height)
        if largest <= max_dimension:
            return image
        scale = max_dimension / float(largest)
        resized = image.resize(
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
        return resized

    def primary_image_path(self, spec: MediaAssetSpec) -> Path | None:
        for component in spec.component_paths:
            path = Path(component)
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                return path
        path = Path(spec.display_path)
        return path if path.suffix.lower() in IMAGE_EXTENSIONS else None

    def primary_video_path(self, spec: MediaAssetSpec) -> Path | None:
        for component in spec.component_paths:
            path = Path(component)
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                return path
        path = Path(spec.display_path)
        return path if path.suffix.lower() in VIDEO_EXTENSIONS else None

    def crop_signature(self, image: Image.Image, bbox: list[int]) -> str | None:
        crop = self.crop_region(image, bbox)
        if crop is None:
            return None
        if imagehash is not None:
            return str(imagehash.average_hash(crop))

        grayscale = crop.convert("L").resize((8, 8))
        pixels = np.asarray(grayscale, dtype=np.float32)
        threshold = float(pixels.mean())
        bits = "".join("1" if value >= threshold else "0" for value in pixels.flatten())
        return f"{int(bits, 2):016x}"

    def crop_region(self, image: Image.Image, bbox: list[int]) -> Image.Image | None:
        x, y, width, height = bbox
        if width <= 1 or height <= 1:
            return None
        x2 = x + width
        y2 = y + height
        crop = image.crop((x, y, x2, y2))
        if crop.size[0] <= 1 or crop.size[1] <= 1:
            return None
        return crop

    def _load_cat_face_detector(self):
        if cv2 is None:
            return None

        for cascade_name in (
            "haarcascade_frontalcatface_extended.xml",
            "haarcascade_frontalcatface.xml",
        ):
            try:
                cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)
            except Exception:
                continue
            if cascade is not None and not cascade.empty():
                return cascade
        return None

    def _available_onnx_providers(self) -> list[str]:
        available: list[str] = []
        if ort is not None:
            try:
                available = list(ort.get_available_providers())
            except Exception:
                available = []
        return available

    def _preferred_recognizer_provider_specs(self) -> list[Any]:
        available = self._available_onnx_providers()
        providers: list[Any] = []
        if "CUDAExecutionProvider" in available:
            self._prepare_onnxruntime_cuda()
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                )
            )
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        if not providers:
            providers = ["CPUExecutionProvider"]
        return providers

    def _preferred_detector_provider_specs(
        self,
        detector_path: Path,
        *,
        detection_input_size: tuple[int, int],
    ) -> list[Any]:
        available = self._available_onnx_providers()
        providers: list[Any] = []
        prefer_tensorrt = self.config.human_face_detector_backend in {"auto", "tensorrt"}
        if (
            prefer_tensorrt
            and "TensorrtExecutionProvider" in available
            and self._tensorrt_runtime_available()
        ):
            self._prepare_onnxruntime_cuda()
            trt_options: dict[str, object] = {
                "device_id": 0,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(self._tensorrt_cache_path()),
                "trt_timing_cache_enable": True,
            }
            detector_input_name = self._inspect_onnx_input_name(detector_path)
            if detector_input_name:
                width, height = detection_input_size
                trt_options["trt_profile_min_shapes"] = f"{detector_input_name}:1x3x{height}x{width}"
                trt_options["trt_profile_opt_shapes"] = f"{detector_input_name}:8x3x{height}x{width}"
                trt_options["trt_profile_max_shapes"] = f"{detector_input_name}:32x3x{height}x{width}"
            providers.append(("TensorrtExecutionProvider", trt_options))
        if "CUDAExecutionProvider" in available:
            self._prepare_onnxruntime_cuda()
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                )
            )
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        if not providers:
            providers = ["CPUExecutionProvider"]
        return providers

    def _inspect_onnx_input_name(self, model_path: Path) -> str:
        if onnx is not None:
            try:
                model = onnx.load(str(model_path), load_external_data=False)
                if model.graph.input:
                    return str(model.graph.input[0].name)
            except Exception:
                pass
        if ort is not None:
            try:
                session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                inputs = session.get_inputs()
                if inputs:
                    return str(inputs[0].name)
            except Exception:
                return ""
        return ""

    def _tensorrt_cache_path(self) -> Path:
        cache_root = self.config.cache_path.parent / "onnxruntime-trt"
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root

    def _tensorrt_runtime_available(self) -> bool:
        required_libraries = (
            "libnvinfer.so.10",
            "libnvinfer_plugin.so.10",
        )
        for library_name in required_libraries:
            if self._can_load_shared_library(library_name):
                continue
            generic_name = library_name.split(".so", 1)[0].removeprefix("lib")
            resolved_name = ctypes.util.find_library(generic_name)
            if resolved_name and self._can_load_shared_library(resolved_name):
                continue
            return False
        return True

    def _can_load_shared_library(self, library_name: str) -> bool:
        try:
            ctypes.CDLL(library_name)
            return True
        except OSError:
            return False

    def _prepare_onnxruntime_cuda(self) -> None:
        if ort is None:
            return
        set_severity = getattr(ort, "set_default_logger_severity", None)
        if callable(set_severity):
            try:
                set_severity(3)
            except Exception:
                pass
        preload = getattr(ort, "preload_dlls", None)
        if not callable(preload):
            return
        try:
            preload(directory="")
        except TypeError:
            try:
                preload()
            except Exception:
                return
        except Exception:
            try:
                preload()
            except Exception:
                return

    def _preferred_yolo_device(self) -> int | str:
        if torch is not None and torch.cuda.is_available():
            return 0
        return "cpu"

    def _yolo_device_label(self) -> str:
        return f"cuda:{self.yolo_device}" if isinstance(self.yolo_device, int) else str(self.yolo_device)

    def _analysis_batch_size(self) -> int:
        configured = max(1, int(self.config.analysis_batch_size))
        if isinstance(self.yolo_device, int):
            return min(configured, 48)
        return min(configured, 12)

    def _pet_embedding_batch_size(self) -> int:
        return max(1, int(self.config.pet_embedding_batch_size))

    def _iter_batches(self, values: list[Any], batch_size: int):
        for start in range(0, len(values), batch_size):
            yield values[start : start + batch_size]

    def _load_human_face_backend(self):
        if ort is None or cv2 is None or not self.config.face_recognition_enabled:
            return None

        detector_path = self._resolve_human_face_detector_path(download_missing=self.config.auto_download_models)
        recognizer_path = self._resolve_human_face_recognizer_path(download_missing=self.config.auto_download_models)
        if detector_path is None or recognizer_path is None:
            return None

        try:
            detection_input_size = (640, 640)
            detector_providers = self._preferred_detector_provider_specs(
                detector_path,
                detection_input_size=detection_input_size,
            )
            recognizer_providers = self._preferred_recognizer_provider_specs()
            backend = HumanFaceBackend(
                detector_path=detector_path,
                recognizer_path=recognizer_path,
                detector_providers=detector_providers,
                recognizer_providers=recognizer_providers,
                detection_input_size=detection_input_size,
            )
            self.human_face_detector_name = detector_path.name
            self.human_face_recognizer_name = recognizer_path.name
            self.human_face_pipeline_id = self.human_face_pipeline_revision(
                detector_name=self.human_face_detector_name,
                recognizer_name=self.human_face_recognizer_name,
            )
            provider_order = list(backend.detector_providers)
            for provider in backend.recognizer_providers:
                if provider not in provider_order:
                    provider_order.append(provider)
            self.human_face_providers = provider_order
            self.human_face_uses_gpu = backend.uses_gpu
            self.human_face_device_label = backend.device_label
            return backend
        except Exception:
            self.human_face_providers = []
            self.human_face_uses_gpu = False
            self.human_face_device_label = ""
            self.human_face_detector_name = ""
            self.human_face_recognizer_name = ""
            self.human_face_pipeline_id = self.human_face_pipeline_revision(self.config)
            return None

    def _resolve_human_face_detector_path(self, *, download_missing: bool) -> Path | None:
        explicit_path = Path(self.config.human_face_detector_path).expanduser() if self.config.human_face_detector_path else None
        if explicit_path and explicit_path.exists():
            return explicit_path.resolve()

        managed_dir = self._ensure_model_path(
            self.config.human_face_detector_model_id,
            download_missing=download_missing,
        )
        if managed_dir is not None:
            detector_path = self._resolve_detector_from_model_path(managed_dir)
            if detector_path is not None:
                return detector_path

        recognizer_pack_dir = self._ensure_model_path(
            self.config.human_face_model_id,
            download_missing=download_missing,
        )
        if recognizer_pack_dir is not None:
            detector_path = self._resolve_detector_from_model_path(recognizer_pack_dir)
            if detector_path is not None:
                return detector_path
        return None

    def _resolve_human_face_recognizer_path(self, *, download_missing: bool) -> Path | None:
        managed_dir = self._ensure_model_path(
            self.config.human_face_model_id,
            download_missing=download_missing,
        )
        if managed_dir is None:
            return None
        for candidate_name in ("w600k_mbf.onnx", "glintr100.onnx"):
            candidate_path = managed_dir / candidate_name
            if candidate_path.exists():
                return candidate_path.resolve()
        return None

    def _ensure_model_path(self, model_id: str, *, download_missing: bool) -> Path | None:
        resolved_path: Path | None = None
        if download_missing:
            managed_path = self.model_manager.ensure_model(model_id)
            if managed_path:
                resolved_path = Path(managed_path).expanduser()
        if resolved_path is not None and resolved_path.exists():
            return resolved_path.resolve()
        status = self.model_manager.status(model_id)
        if status.installed:
            path = Path(status.local_path).expanduser()
            if path.exists():
                return path.resolve()
        return None

    def _resolve_detector_from_model_path(self, model_path: Path) -> Path | None:
        if model_path.is_file():
            return model_path.resolve()
        for candidate_name in ("scrfd_10g_bnkps.onnx", "det_500m.onnx"):
            candidate_path = model_path / candidate_name
            if candidate_path.exists():
                return candidate_path.resolve()
        return None

    def _load_yolo_model(self, model_id: str, configured_path: str = ""):
        if YOLO is None:
            return None

        model_path = ""
        explicit_path = Path(configured_path).expanduser() if configured_path else None
        if explicit_path and explicit_path.exists():
            model_path = str(explicit_path)
        else:
            managed_path = self.model_manager.ensure_model(model_id)
            if managed_path:
                model_path = managed_path
            else:
                local_status = self.model_manager.status(model_id)
                if local_status.installed:
                    model_path = local_status.local_path

        if not model_path:
            return None

        try:
            return YOLO(model_path)
        except Exception:
            return None

    def _load_object_model(self):
        if not self.config.object_detection_enabled:
            return None
        return self._load_yolo_model(
            self.config.object_model_id,
            configured_path=self.config.object_model_path,
        )

    def _load_pet_face_model(self):
        if not self.config.pet_recognition_enabled:
            return None
        return self._load_yolo_model(self.config.pet_detector_model_id)

    def _load_pet_embedding_model(self):
        if not self.config.pet_recognition_enabled:
            return None

        model_dir = self.model_manager.ensure_model(self.config.pet_embedding_model_id)
        if not model_dir:
            local_status = self.model_manager.status(self.config.pet_embedding_model_id)
            if local_status.installed:
                model_dir = local_status.local_path

        if not model_dir:
            return None

        try:
            return PetEmbeddingModel(Path(model_dir))
        except Exception:
            return None

    @staticmethod
    def human_face_pipeline_revision(
        config: AppConfig | None = None,
        *,
        detector_name: str = "",
        recognizer_name: str = "",
    ) -> str:
        if detector_name or recognizer_name:
            detector_token = detector_name or "none"
            recognizer_token = recognizer_name or "none"
            return f"{HUMAN_FACE_PIPELINE_VERSION}:{detector_token}:{recognizer_token}"
        if config is None:
            return HUMAN_FACE_PIPELINE_VERSION
        detector_token = (
            Path(config.human_face_detector_path).expanduser().name
            if config.human_face_detector_path
            else config.human_face_detector_model_id
        )
        recognizer_token = config.human_face_model_id
        return f"{HUMAN_FACE_PIPELINE_VERSION}:{detector_token}:{recognizer_token}"

    def _detect_human_faces(self, image: Image.Image) -> list[DetectionRegion]:
        return self._detect_human_faces_batch([image])[0]

    def _detect_human_faces_batch(
        self,
        images: list[Image.Image],
    ) -> list[list[DetectionRegion]]:
        if not images:
            return []
        if self.human_face_backend is None or cv2 is None:
            return [[] for _ in images]

        bgr_images = [cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) for image in images]
        try:
            candidates_by_image = self.human_face_backend.detect_faces_batch(bgr_images)
            embeddings_by_image = self.human_face_backend.embed_faces_batch(bgr_images, candidates_by_image)
        except Exception:
            return [[] for _ in images]

        outputs: list[list[DetectionRegion]] = [[] for _ in images]
        for image_index, image in enumerate(images):
            candidates = candidates_by_image[image_index] if image_index < len(candidates_by_image) else []
            embeddings = embeddings_by_image[image_index] if image_index < len(embeddings_by_image) else []
            detections: list[DetectionRegion] = []
            for detection_index, candidate in enumerate(candidates):
                x1, y1, x2, y2 = [int(round(value)) for value in candidate.bbox_xyxy]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.size[0], x2)
                y2 = min(image.size[1], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox = [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]
                embedding = embeddings[detection_index] if detection_index < len(embeddings) else []
                detections.append(
                    DetectionRegion(
                        id=f"face-{detection_index}",
                        kind="face",
                        label="face",
                        confidence=float(candidate.confidence),
                        bbox=bbox,
                        encoding=embedding,
                        signature=self.crop_signature(image, bbox),
                    )
                )
            outputs[image_index] = detections
        return outputs

    def _detect_objects(self, image: Image.Image) -> tuple[list[DetectionRegion], set[str]]:
        return self._detect_objects_batch([image])[0]

    def _detect_objects_batch(
        self,
        images: list[Image.Image],
    ) -> list[tuple[list[DetectionRegion], set[str]]]:
        if not images:
            return []
        if self.object_model is None:
            return [([], set()) for _ in images]

        outputs: list[tuple[list[DetectionRegion], set[str]]] = []
        for image_batch in self._iter_batches(images, self._analysis_batch_size()):
            rgb_batch = [np.asarray(image) for image in image_batch]
            try:
                results = self.object_model.predict(
                    rgb_batch,
                    device=self.yolo_device,
                    half=isinstance(self.yolo_device, int),
                    verbose=False,
                )
            except Exception:
                outputs.extend([([], set()) for _ in image_batch])
                continue

            for image, result in zip(image_batch, results, strict=False):
                outputs.append(self._parse_object_result(image, result))

            missing_results = len(image_batch) - len(results)
            if missing_results > 0:
                outputs.extend([([], set()) for _ in range(missing_results)])
        return outputs

    def _parse_object_result(
        self,
        image: Image.Image,
        result: Any,
    ) -> tuple[list[DetectionRegion], set[str]]:
        detections: list[DetectionRegion] = []
        tags: set[str] = set()
        names = getattr(result, "names", getattr(self.object_model, "names", {}))
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return [], set()

        for index, box in enumerate(boxes):
            confidence = float(box.conf[0])
            label_index = int(box.cls[0])
            label = str(names.get(label_index, label_index)).replace("_", " ").lower()
            if confidence < self._object_threshold(label):
                continue
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            bbox = [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]
            detections.append(
                DetectionRegion(
                    id=f"object-{index}",
                    kind="object",
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    signature=self.crop_signature(image, bbox),
                )
            )
            tags.add(label)

        return detections, tags

    def _object_threshold(self, label: str) -> float:
        normalized = label.lower()
        if normalized in CAT_LABELS:
            return 0.18
        if normalized in DOG_LABELS:
            return 0.24
        if normalized in PET_LABELS:
            return 0.28
        return 0.35

    def _build_pet_detections(
        self,
        image: Image.Image,
        object_detections: list[DetectionRegion],
    ) -> tuple[list[DetectionRegion], set[str]]:
        pet_face_candidates = self._detect_pet_face_candidates_batch([image])[0]
        return self._build_pet_detections_from_candidates(
            image,
            object_detections,
            pet_face_candidates,
            embed=True,
        )

    def _build_pet_detections_from_candidates(
        self,
        image: Image.Image,
        object_detections: list[DetectionRegion],
        pet_face_candidates: list[PetFaceCandidate],
        embed: bool,
    ) -> tuple[list[DetectionRegion], set[str]]:
        pet_detections: list[DetectionRegion] = []
        pet_tags: set[str] = set()
        pet_face_detections = self._build_pet_face_detections_from_candidates(
            image,
            object_detections,
            pet_face_candidates,
            embed=embed,
        )
        pet_detections.extend(pet_face_detections)
        for detection in pet_face_detections:
            pet_tags.update(self._pet_tags_for_label(detection.label, face_detection=True))

        cat_object_detections = [
            detection for detection in object_detections if detection.label.lower() in CAT_LABELS
        ]
        has_cat_face_detection = any(
            detection.kind == "pet_face" and detection.label.lower() in CAT_LABELS
            for detection in pet_face_detections
        )

        for detection in object_detections:
            label = detection.label.lower()
            if label not in PET_LABELS:
                continue

            if label in CAT_LABELS:
                if self._has_overlapping_pet_face(detection.bbox, pet_face_detections):
                    pet_tags.update({"animal", "pet", "cat", "cat face"})
                    continue
                cat_faces = self._detect_cat_faces(image, detection.bbox)
                if cat_faces:
                    for cat_index, bbox in enumerate(cat_faces):
                        pet_detections.append(
                            self._build_pet_region(
                                image=image,
                                detection_id=f"pet-cat-{detection.id}-{cat_index}",
                                label="cat",
                                bbox=bbox,
                                confidence=max(detection.confidence, 0.55),
                                kind="pet_face",
                                embed=embed,
                            )
                        )
                    pet_tags.update({"animal", "pet", "cat", "cat face"})
                    continue
                pet_tags.update({"animal", "pet", "cat"})
                pet_detections.append(
                    self._build_pet_region(
                        image=image,
                        detection_id=f"pet-{detection.id}",
                        label="cat",
                        bbox=detection.bbox,
                        confidence=detection.confidence,
                        kind="pet",
                        embed=embed,
                    )
                )
                continue

            if label in DOG_LABELS and self._has_overlapping_pet_face(detection.bbox, pet_face_detections):
                pet_tags.update({"animal", "pet", "dog", "dog face"})
                continue

            pet_detections.append(
                self._build_pet_region(
                    image=image,
                    detection_id=f"pet-{detection.id}",
                    label=label,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    kind="pet",
                    embed=embed,
                )
            )
            pet_tags.update({"animal", "pet", label})

        if not cat_object_detections and not has_cat_face_detection:
            for cat_index, bbox in enumerate(self._detect_cat_faces(image)):
                if self._has_overlapping_pet_face(bbox, pet_face_detections):
                    continue
                pet_detections.append(
                    self._build_pet_region(
                        image=image,
                        detection_id=f"pet-cat-fallback-{cat_index}",
                        label="cat",
                        bbox=bbox,
                        confidence=0.52,
                        kind="pet_face",
                        embed=embed,
                    )
                )
                pet_tags.update({"animal", "pet", "cat", "cat face"})

        return pet_detections, pet_tags

    def _detect_pet_faces(
        self,
        image: Image.Image,
        object_detections: list[DetectionRegion],
    ) -> list[DetectionRegion]:
        pet_face_candidates = self._detect_pet_face_candidates_batch([image])[0]
        return self._build_pet_face_detections_from_candidates(
            image,
            object_detections,
            pet_face_candidates,
            embed=True,
        )

    def _detect_pet_face_candidates_batch(
        self,
        images: list[Image.Image],
    ) -> list[list[PetFaceCandidate]]:
        if not images:
            return []
        if self.pet_face_model is None:
            return [[] for _ in images]

        outputs: list[list[PetFaceCandidate]] = []
        for image_batch in self._iter_batches(images, self._analysis_batch_size()):
            rgb_batch = [np.asarray(image) for image in image_batch]
            try:
                results = self.pet_face_model.predict(
                    rgb_batch,
                    device=self.yolo_device,
                    half=isinstance(self.yolo_device, int),
                    verbose=False,
                )
            except Exception:
                outputs.extend([[] for _ in image_batch])
                continue

            for result in results:
                outputs.append(self._parse_pet_face_candidates(result))

            missing_results = len(image_batch) - len(results)
            if missing_results > 0:
                outputs.extend([[] for _ in range(missing_results)])
        return outputs

    def _parse_pet_face_candidates(self, result: Any) -> list[PetFaceCandidate]:
        names = getattr(result, "names", getattr(self.pet_face_model, "names", {}))
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        candidates: list[PetFaceCandidate] = []
        for box in boxes:
            confidence = float(box.conf[0])
            label_index = int(box.cls[0])
            raw_label = str(names.get(label_index, label_index)).replace("_", " ").lower()
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            bbox = [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]
            candidates.append(
                PetFaceCandidate(
                    label=raw_label,
                    confidence=confidence,
                    bbox=bbox,
                )
            )
        return candidates

    def _build_pet_face_detections_from_candidates(
        self,
        image: Image.Image,
        object_detections: list[DetectionRegion],
        pet_face_candidates: list[PetFaceCandidate],
        embed: bool,
    ) -> list[DetectionRegion]:
        detections: list[DetectionRegion] = []
        for index, candidate in enumerate(pet_face_candidates):
            label = self._normalize_pet_face_label(candidate.label, image, candidate.bbox, object_detections)
            if candidate.confidence < self._pet_face_threshold(label or candidate.label):
                continue
            detections.append(
                self._build_pet_region(
                    image=image,
                    detection_id=f"pet-face-{index}",
                    label=label or "pet",
                    bbox=candidate.bbox,
                    confidence=candidate.confidence,
                    kind="pet_face",
                    embed=embed,
                )
            )
        return detections

    def _build_pet_region(
        self,
        image: Image.Image,
        detection_id: str,
        label: str,
        bbox: list[int],
        confidence: float,
        kind: str,
        embed: bool = True,
    ) -> DetectionRegion:
        crop = self.crop_region(image, bbox)
        embedding = self._embed_pet_crop(crop) if crop is not None and embed else []
        return DetectionRegion(
            id=detection_id,
            kind=kind,
            label=label,
            confidence=confidence,
            bbox=bbox,
            encoding=embedding,
            signature=self.crop_signature(image, bbox),
        )

    def _embed_pet_crop(self, crop: Image.Image | None) -> list[float]:
        if crop is None:
            return []
        embeddings = self._embed_pet_crops_batch([crop])
        return embeddings[0] if embeddings else []

    def _embed_pet_crops_batch(self, crops: list[Image.Image]) -> list[list[float]]:
        if not crops or self.pet_embedding_model is None:
            return [[] for _ in crops]

        outputs: list[list[float]] = []
        for crop_batch in self._iter_batches(crops, self._pet_embedding_batch_size()):
            try:
                outputs.extend(self.pet_embedding_model.embed_batch(crop_batch))
            except Exception:
                outputs.extend([[] for _ in crop_batch])
        return outputs

    def _normalize_pet_face_label(
        self,
        raw_label: str,
        image: Image.Image,
        bbox: list[int],
        object_detections: list[DetectionRegion],
    ) -> str:
        normalized = raw_label.strip().lower().replace("-", " ")
        if normalized in CAT_LABELS or "cat" in normalized or "kitten" in normalized:
            return "cat"
        if normalized in DOG_LABELS or "dog" in normalized or "puppy" in normalized:
            return "dog"

        overlaps = [
            detection
            for detection in object_detections
            if detection.label.lower() in CAT_LABELS | DOG_LABELS
            and self._iou(bbox, detection.bbox) >= 0.12
        ]
        if any(detection.label.lower() in CAT_LABELS for detection in overlaps):
            return "cat"
        if any(detection.label.lower() in DOG_LABELS for detection in overlaps):
            return "dog"
        if self._detect_cat_faces(image, bbox):
            return "cat"
        return ""

    def _pet_face_threshold(self, label: str) -> float:
        normalized = label.lower()
        if normalized in CAT_LABELS:
            return 0.14
        if normalized in DOG_LABELS:
            return 0.18
        return 0.22

    def _pet_tags_for_label(self, label: str, face_detection: bool = False) -> set[str]:
        normalized = label.lower()
        tags = {"animal", "pet"}
        if normalized:
            tags.add(normalized)
        if face_detection:
            if normalized in CAT_LABELS:
                tags.add("cat face")
            elif normalized in DOG_LABELS:
                tags.add("dog face")
            else:
                tags.add("pet face")
        return tags

    def _has_overlapping_pet_face(
        self,
        bbox: list[int],
        pet_face_detections: list[DetectionRegion],
    ) -> bool:
        for detection in pet_face_detections:
            if detection.kind != "pet_face":
                continue
            if self._iou(bbox, detection.bbox) >= 0.12:
                return True
        return False

    def _iou(self, left: list[int], right: list[int]) -> float:
        left_x1, left_y1, left_w, left_h = left
        right_x1, right_y1, right_w, right_h = right
        left_x2 = left_x1 + left_w
        left_y2 = left_y1 + left_h
        right_x2 = right_x1 + right_w
        right_y2 = right_y1 + right_h

        inter_x1 = max(left_x1, right_x1)
        inter_y1 = max(left_y1, right_y1)
        inter_x2 = min(left_x2, right_x2)
        inter_y2 = min(left_y2, right_y2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        if intersection <= 0:
            return 0.0

        left_area = max(1, left_w * left_h)
        right_area = max(1, right_w * right_h)
        union = left_area + right_area - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _detect_cat_faces(
        self,
        image: Image.Image,
        search_bbox: list[int] | None = None,
    ) -> list[list[int]]:
        if self.cat_face_detector is None or cv2 is None:
            return []

        if search_bbox:
            crop = self.crop_region(image, search_bbox)
            if crop is None:
                return []
            offset_x, offset_y = search_bbox[0], search_bbox[1]
        else:
            crop = image
            offset_x = 0
            offset_y = 0

        gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
        faces = self.cat_face_detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(40, 40),
        )

        detections: list[list[int]] = []
        for x, y, width, height in faces:
            detections.append(
                [int(x + offset_x), int(y + offset_y), int(width), int(height)]
            )
        return detections
