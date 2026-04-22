from __future__ import annotations

from dataclasses import dataclass
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

from ..config import AppConfig
from ..media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, MediaAssetSpec
from ..models import DetectionRegion
from .model_manager import ModelManager

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

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


@dataclass(slots=True)
class AnalysisResult:
    tags: list[str]
    detections: list[DetectionRegion]
    metadata: dict[str, Any]


class PetEmbeddingModel:
    def __init__(self, model_dir: Path) -> None:
        if torch is None or AutoModel is None or AutoImageProcessor is None:
            raise RuntimeError("Pet embedding dependencies are not installed.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(str(model_dir), use_fast=True)
        self.model = AutoModel.from_pretrained(str(model_dir)).to(self.device).eval()

    def embed(self, image: Image.Image) -> list[float]:
        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        if hasattr(outputs, "last_hidden_state"):
            tensor = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output"):
            tensor = outputs.pooler_output
        else:
            raise RuntimeError("Pet embedding model returned an unsupported output structure.")

        tensor = torch.nn.functional.normalize(tensor, dim=1)
        vector = tensor.detach().cpu().numpy()[0]
        return [round(float(value), 6) for value in vector.tolist()]


class VisionAnalyzer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model_manager = ModelManager(config)
        self.cat_face_detector = self._load_cat_face_detector()
        self.human_face_app = self._load_human_face_backend()
        self.object_model = self._load_object_model()
        self.pet_face_model = self._load_pet_face_model()
        self.pet_embedding_model = self._load_pet_embedding_model()

    def analyze(self, spec: MediaAssetSpec) -> AnalysisResult:
        if spec.media_kind == "video":
            video_result = self._analyze_video(spec)
            if video_result.tags or video_result.detections or video_result.metadata.get("video_ai_frames_analyzed"):
                return video_result

        if spec.media_kind == "live_photo":
            still_image = self.load_preview_image(spec)
            still_result = self._analyze_still_image(still_image) if still_image is not None else self._empty_analysis()
            video_result = self._analyze_video(spec)
            if video_result.tags or video_result.detections or video_result.metadata.get("video_ai_frames_analyzed"):
                return self._merge_analysis_results(still_result, video_result)
            return still_result

        image = self.load_preview_image(spec)
        return self._analyze_still_image(image)

    def _empty_analysis(self) -> AnalysisResult:
        return AnalysisResult(tags=[], detections=[], metadata={})

    def _analyze_still_image(self, image: Image.Image | None) -> AnalysisResult:
        if image is None:
            return self._empty_analysis()

        tags: set[str] = set()
        detections: list[DetectionRegion] = []

        human_face_detections = self._detect_human_faces(image)
        if human_face_detections:
            tags.update({"face", "person"})
            detections.extend(human_face_detections)

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
            "human_face_model": "insightface" if self.human_face_app else "",
            "object_model": self.config.object_model_id if self.object_model else "",
            "pet_face_model": self.config.pet_detector_model_id if self.pet_face_model else "",
            "pet_embedding_model": self.config.pet_embedding_model_id if self.pet_embedding_model else "",
            "cat_face_fallback": bool(self.cat_face_detector),
        }

    def _analyze_video(self, spec: MediaAssetSpec) -> AnalysisResult:
        if cv2 is None or not self.config.video_ai_enabled:
            return self._empty_analysis()

        video_path = self.primary_video_path(spec)
        if not video_path:
            return self._empty_analysis()

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return self._empty_analysis()

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            sample_indices = self._build_video_sample_indices(frame_count, fps)
            tags: set[str] = set()
            detections: list[DetectionRegion] = []
            sampled_timestamps: list[float] = []
            analyzed_image: Image.Image | None = None

            for sample_number, frame_index in enumerate(sample_indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                frame_result = self._analyze_still_image(image)
                frame_detections = [
                    detection
                    for detection in frame_result.detections
                    if detection.kind == "face" or self._is_pet_detection(detection)
                ]
                timestamp_seconds = frame_index / fps if fps > 0 else float(sample_number)
                tags.update(frame_result.tags)
                self._merge_unique_detections(
                    detections,
                    self._prefix_detections(
                        frame_detections,
                        prefix=f"video-{sample_number:02d}-{timestamp_seconds:07.2f}",
                    ),
                )
                sampled_timestamps.append(round(timestamp_seconds, 2))
                if analyzed_image is None:
                    analyzed_image = image

            if analyzed_image is None:
                return self._empty_analysis()

            metadata = self._analysis_metadata(analyzed_image)
            metadata.update(
                {
                    "video_ai_frames_analyzed": len(sampled_timestamps),
                    "video_ai_sample_timestamps": sampled_timestamps,
                    "video_ai_mode": "interval_sampling",
                    "video_ai_frame_count": frame_count,
                    "video_ai_fps": round(fps, 3) if fps > 0 else 0.0,
                }
            )
            return AnalysisResult(tags=sorted(tags), detections=detections, metadata=metadata)
        finally:
            capture.release()

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
        try:
            if spec.media_kind in {"image", "live_photo"}:
                image_path = self.primary_image_path(spec)
                if image_path:
                    with Image.open(image_path) as image:
                        return ImageOps.exif_transpose(image).convert("RGB")
                return None

            if spec.media_kind == "gif":
                with Image.open(spec.display_path) as image:
                    frame = next(ImageSequence.Iterator(image))
                    return ImageOps.exif_transpose(frame).convert("RGB")

            video_path = self.primary_video_path(spec)
            if not video_path or cv2 is None:
                return None

            capture = cv2.VideoCapture(str(video_path))
            try:
                ok, frame = capture.read()
                if not ok or frame is None:
                    return None
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_frame)
            finally:
                capture.release()
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            return None

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

    def _load_human_face_backend(self):
        if FaceAnalysis is None or not self.config.face_recognition_enabled:
            return None

        pack_root = self.config.models_path / "insightface"
        pack_root.mkdir(parents=True, exist_ok=True)
        pack_name = self._insightface_pack_name()
        pack_dir = pack_root / "models" / pack_name

        if not pack_dir.exists():
            if self.config.auto_download_models:
                self.model_manager.download_model(self.config.human_face_model_id)
            else:
                return None

        try:
            app = FaceAnalysis(
                name=pack_name,
                root=str(pack_root),
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            return app
        except Exception:
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

    def _insightface_pack_name(self) -> str:
        if self.config.human_face_model_id == "insightface_buffalo_sc":
            return "buffalo_sc"
        return "buffalo_sc"

    def _detect_human_faces(self, image: Image.Image) -> list[DetectionRegion]:
        if self.human_face_app is None or cv2 is None:
            return []

        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        try:
            faces = self.human_face_app.get(bgr)
        except Exception:
            return []

        detections: list[DetectionRegion] = []
        for index, face in enumerate(faces):
            bbox_float = getattr(face, "bbox", None)
            if bbox_float is None:
                continue
            x1, y1, x2, y2 = [int(value) for value in bbox_float]
            bbox = [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]
            embedding_raw = getattr(face, "normed_embedding", None)
            if embedding_raw is None:
                embedding_raw = getattr(face, "embedding", None)
            embedding = []
            if embedding_raw is not None:
                embedding = [round(float(value), 6) for value in np.asarray(embedding_raw).tolist()]
            detections.append(
                DetectionRegion(
                    id=f"face-{index}",
                    kind="face",
                    label="face",
                    confidence=float(getattr(face, "det_score", 1.0)),
                    bbox=bbox,
                    encoding=embedding,
                    signature=self.crop_signature(image, bbox),
                )
            )
        return detections

    def _detect_objects(self, image: Image.Image) -> tuple[list[DetectionRegion], set[str]]:
        if self.object_model is None:
            return [], set()

        rgb = np.asarray(image)
        try:
            results = self.object_model.predict(rgb, verbose=False)
        except Exception:
            return [], set()

        detections: list[DetectionRegion] = []
        tags: set[str] = set()
        result = results[0]
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
        pet_detections: list[DetectionRegion] = []
        pet_tags: set[str] = set()
        pet_face_detections = self._detect_pet_faces(image, object_detections)
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
                    )
                )
                pet_tags.update({"animal", "pet", "cat", "cat face"})

        return pet_detections, pet_tags

    def _detect_pet_faces(
        self,
        image: Image.Image,
        object_detections: list[DetectionRegion],
    ) -> list[DetectionRegion]:
        if self.pet_face_model is None:
            return []

        rgb = np.asarray(image)
        try:
            results = self.pet_face_model.predict(rgb, verbose=False)
        except Exception:
            return []

        detections: list[DetectionRegion] = []
        result = results[0]
        names = getattr(result, "names", getattr(self.pet_face_model, "names", {}))
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        for index, box in enumerate(boxes):
            confidence = float(box.conf[0])
            label_index = int(box.cls[0])
            raw_label = str(names.get(label_index, label_index)).replace("_", " ").lower()
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            bbox = [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]
            label = self._normalize_pet_face_label(raw_label, image, bbox, object_detections)
            if confidence < self._pet_face_threshold(label or raw_label):
                continue
            detections.append(
                self._build_pet_region(
                    image=image,
                    detection_id=f"pet-face-{index}",
                    label=label or "pet",
                    bbox=bbox,
                    confidence=confidence,
                    kind="pet_face",
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
    ) -> DetectionRegion:
        crop = self.crop_region(image, bbox)
        embedding = self._embed_pet_crop(crop) if crop is not None else []
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
        if crop is None or self.pet_embedding_model is None:
            return []
        try:
            return self.pet_embedding_model.embed(crop)
        except Exception:
            return []

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
