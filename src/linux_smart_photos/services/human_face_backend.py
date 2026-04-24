from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None

import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


HUMAN_FACE_PIPELINE_VERSION = "scrfd_arcface_onnx_v3"

ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass(slots=True)
class HumanFaceDetection:
    bbox_xyxy: list[float]
    confidence: float
    keypoints: np.ndarray | None


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for index in range(0, distance.shape[1], 2):
        px = points[:, index % 2] + distance[:, index]
        py = points[:, index % 2 + 1] + distance[:, index + 1]
        predictions.append(px)
        predictions.append(py)
    return np.stack(predictions, axis=-1)


class HumanFaceBackend:
    def __init__(
        self,
        *,
        detector_path: Path,
        recognizer_path: Path,
        detector_providers: list[Any],
        recognizer_providers: list[Any],
        detection_input_size: tuple[int, int] = (640, 640),
        detection_threshold: float = 0.48,
        nms_threshold: float = 0.4,
    ) -> None:
        if ort is None or cv2 is None:
            raise RuntimeError("Human face backend dependencies are unavailable.")

        self.detector_session = ort.InferenceSession(str(detector_path), providers=detector_providers)
        self.recognizer_session = ort.InferenceSession(str(recognizer_path), providers=recognizer_providers)
        self.detector_providers = list(self.detector_session.get_providers())
        self.recognizer_providers = list(self.recognizer_session.get_providers())
        self.detector_uses_gpu = any(
            provider in {"TensorrtExecutionProvider", "CUDAExecutionProvider"}
            for provider in self.detector_providers
        )
        self.recognizer_uses_gpu = any(
            provider in {"TensorrtExecutionProvider", "CUDAExecutionProvider"}
            for provider in self.recognizer_providers
        )
        self.uses_gpu = self.detector_uses_gpu or self.recognizer_uses_gpu
        if "TensorrtExecutionProvider" in self.detector_providers:
            self.detector_device_label = "tensorrt"
        elif self.detector_uses_gpu:
            self.detector_device_label = "cuda"
        else:
            self.detector_device_label = "cpu"
        if "TensorrtExecutionProvider" in self.recognizer_providers:
            self.recognizer_device_label = "tensorrt"
        elif self.recognizer_uses_gpu:
            self.recognizer_device_label = "cuda"
        else:
            self.recognizer_device_label = "cpu"
        if "TensorrtExecutionProvider" in self.detector_providers:
            self.device_label = "tensorrt"
        elif self.uses_gpu:
            self.device_label = "cuda"
        else:
            self.device_label = "cpu"
        self.detection_threshold = float(detection_threshold)
        self.nms_threshold = float(nms_threshold)
        self.center_cache: dict[tuple[int, int, int], np.ndarray] = {}

        self.detector_input_name = self.detector_session.get_inputs()[0].name
        self.detector_output_names = [output.name for output in self.detector_session.get_outputs()]
        detector_input_shape = self.detector_session.get_inputs()[0].shape
        detector_outputs = self.detector_session.get_outputs()

        if not all(isinstance(value, int) and value > 0 for value in detector_input_shape[2:4]):
            self.detection_input_size = detection_input_size
        else:
            self.detection_input_size = tuple(int(value) for value in detector_input_shape[2:4][::-1])
        self.detector_batchable = (
            len(detector_outputs[0].shape) == 3
            and not (isinstance(detector_input_shape[0], int) and detector_input_shape[0] == 1)
        )

        self.detector_input_mean = 127.5
        self.detector_input_std = 128.0
        self.use_keypoints = False
        self.fmc = 0
        self.feature_strides: list[int] = []
        self.num_anchors = 1
        self._init_detector_layout(detector_outputs)

        recognizer_input = self.recognizer_session.get_inputs()[0]
        recognizer_outputs = self.recognizer_session.get_outputs()
        recognizer_input_shape = recognizer_input.shape
        self.recognizer_input_name = recognizer_input.name
        self.recognizer_output_name = recognizer_outputs[0].name
        self.recognizer_input_size = tuple(
            int(value) if isinstance(value, int) and value > 0 else 112
            for value in recognizer_input_shape[2:4][::-1]
        )
        self.recognizer_batchable = not (
            isinstance(recognizer_input_shape[0], int) and recognizer_input_shape[0] == 1
        )
        self.recognizer_input_mean = 127.5
        self.recognizer_input_std = 127.5
        self.recognizer_batch_size = 192 if self.uses_gpu else 24

    def _init_detector_layout(self, outputs: list[Any]) -> None:
        if len(outputs) == 6:
            self.fmc = 3
            self.feature_strides = [8, 16, 32]
            self.num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self.feature_strides = [8, 16, 32]
            self.num_anchors = 2
            self.use_keypoints = True
        elif len(outputs) == 10:
            self.fmc = 5
            self.feature_strides = [8, 16, 32, 64, 128]
            self.num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self.feature_strides = [8, 16, 32, 64, 128]
            self.num_anchors = 1
            self.use_keypoints = True
        else:
            raise RuntimeError(f"Unsupported SCRFD output layout with {len(outputs)} outputs.")

    def detect_faces_batch(self, images_bgr: list[np.ndarray]) -> list[list[HumanFaceDetection]]:
        if not images_bgr:
            return []

        prepared_inputs = [self._prepare_detector_image(image) for image in images_bgr]
        if self.detector_batchable and len(prepared_inputs) > 1:
            batch_outputs = self._run_detector_batch([prepared[0] for prepared in prepared_inputs])
            return [
                self._decode_detector_outputs(
                    net_outputs=batch_outputs,
                    batch_index=batch_index,
                    input_shape=prepared_inputs[batch_index][0].shape,
                    scale=prepared_inputs[batch_index][1],
                )
                for batch_index in range(len(prepared_inputs))
            ]

        detections: list[list[HumanFaceDetection]] = []
        for det_image, scale in prepared_inputs:
            net_outputs = self._run_detector_single(det_image)
            detections.append(
                self._decode_detector_outputs(
                    net_outputs=net_outputs,
                    batch_index=0,
                    input_shape=det_image.shape,
                    scale=scale,
                )
            )
        return detections

    def embed_faces_batch(
        self,
        images_bgr: list[np.ndarray],
        detections_by_image: list[list[HumanFaceDetection]],
    ) -> list[list[list[float]]]:
        per_image_embeddings: list[list[list[float]]] = [[] for _ in images_bgr]
        aligned_faces: list[np.ndarray] = []
        mapping: list[tuple[int, int]] = []

        for image_index, (image, detections) in enumerate(zip(images_bgr, detections_by_image, strict=False)):
            per_image_embeddings[image_index] = [[] for _ in detections]
            for detection_index, detection in enumerate(detections):
                aligned = self._align_face(image, detection)
                if aligned is None:
                    continue
                aligned_faces.append(aligned)
                mapping.append((image_index, detection_index))

        if not aligned_faces:
            return per_image_embeddings

        embeddings = self._run_recognizer(aligned_faces)
        for (image_index, detection_index), embedding in zip(mapping, embeddings, strict=False):
            per_image_embeddings[image_index][detection_index] = embedding
        return per_image_embeddings

    def _prepare_detector_image(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        input_width, input_height = self.detection_input_size
        image_ratio = float(image_bgr.shape[0]) / float(max(1, image_bgr.shape[1]))
        model_ratio = float(input_height) / float(max(1, input_width))
        if image_ratio > model_ratio:
            new_height = input_height
            new_width = int(new_height / image_ratio)
        else:
            new_width = input_width
            new_height = int(new_width * image_ratio)
        det_scale = float(new_height) / float(max(1, image_bgr.shape[0]))
        resized = cv2.resize(image_bgr, (new_width, new_height))
        det_image = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        det_image[:new_height, :new_width, :] = resized
        return det_image, det_scale

    def _run_detector_batch(self, det_images: list[np.ndarray]) -> list[np.ndarray]:
        blob = cv2.dnn.blobFromImages(
            det_images,
            1.0 / self.detector_input_std,
            self.detection_input_size,
            (self.detector_input_mean, self.detector_input_mean, self.detector_input_mean),
            swapRB=True,
        )
        return self.detector_session.run(
            self.detector_output_names,
            {self.detector_input_name: blob},
        )

    def _run_detector_single(self, det_image: np.ndarray) -> list[np.ndarray]:
        blob = cv2.dnn.blobFromImage(
            det_image,
            1.0 / self.detector_input_std,
            self.detection_input_size,
            (self.detector_input_mean, self.detector_input_mean, self.detector_input_mean),
            swapRB=True,
        )
        return self.detector_session.run(
            self.detector_output_names,
            {self.detector_input_name: blob},
        )

    def _decode_detector_outputs(
        self,
        *,
        net_outputs: list[np.ndarray],
        batch_index: int,
        input_shape: tuple[int, ...],
        scale: float,
    ) -> list[HumanFaceDetection]:
        scores_list: list[np.ndarray] = []
        boxes_list: list[np.ndarray] = []
        keypoints_list: list[np.ndarray] = []
        input_height = int(input_shape[0])
        input_width = int(input_shape[1])

        for level_index, stride in enumerate(self.feature_strides):
            scores = self._select_detector_output(net_outputs[level_index], batch_index).reshape(-1)
            bbox_predictions = self._select_detector_output(net_outputs[level_index + self.fmc], batch_index)
            bbox_predictions = bbox_predictions.reshape((-1, 4)) * stride
            if self.use_keypoints:
                keypoint_predictions = self._select_detector_output(
                    net_outputs[level_index + self.fmc * 2],
                    batch_index,
                ).reshape((-1, 10)) * stride

            height = input_height // stride
            width = input_width // stride
            cache_key = (height, width, stride)
            anchor_centers = self.center_cache.get(cache_key)
            if anchor_centers is None:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self.num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self.num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[cache_key] = anchor_centers

            positive_indices = np.where(scores >= self.detection_threshold)[0]
            if positive_indices.size == 0:
                continue

            boxes = _distance2bbox(anchor_centers, bbox_predictions)[positive_indices]
            boxes_list.append(boxes)
            scores_list.append(scores[positive_indices])
            if self.use_keypoints:
                keypoints = _distance2kps(anchor_centers, keypoint_predictions).reshape((-1, 5, 2))
                keypoints_list.append(keypoints[positive_indices])

        if not scores_list:
            return []

        scores = np.concatenate(scores_list, axis=0)
        boxes = np.concatenate(boxes_list, axis=0) / scale
        order = scores.argsort()[::-1]
        scores = scores[order]
        boxes = boxes[order]
        if self.use_keypoints:
            keypoints = np.concatenate(keypoints_list, axis=0)[order] / scale
        else:
            keypoints = None

        detections = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)
        keep_indices = self._nms(detections)
        detections = detections[keep_indices]
        if keypoints is not None:
            keypoints = keypoints[keep_indices]

        results: list[HumanFaceDetection] = []
        for detection_index, detection in enumerate(detections):
            bbox = [float(value) for value in detection[:4].tolist()]
            landmark = keypoints[detection_index] if keypoints is not None else None
            results.append(
                HumanFaceDetection(
                    bbox_xyxy=bbox,
                    confidence=float(detection[4]),
                    keypoints=landmark,
                )
            )
        return results

    def _select_detector_output(self, output: np.ndarray, batch_index: int) -> np.ndarray:
        if output.ndim >= 3:
            return output[batch_index]
        return output

    def _nms(self, detections: np.ndarray) -> list[int]:
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]
        areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            current_index = int(order[0])
            keep.append(current_index)
            xx1 = np.maximum(x1[current_index], x1[order[1:]])
            yy1 = np.maximum(y1[current_index], y1[order[1:]])
            xx2 = np.minimum(x2[current_index], x2[order[1:]])
            yy2 = np.minimum(y2[current_index], y2[order[1:]])
            width = np.maximum(0.0, xx2 - xx1 + 1.0)
            height = np.maximum(0.0, yy2 - yy1 + 1.0)
            intersection = width * height
            overlap = intersection / (areas[current_index] + areas[order[1:]] - intersection)
            remaining = np.where(overlap <= self.nms_threshold)[0]
            order = order[remaining + 1]

        return keep

    def _align_face(self, image_bgr: np.ndarray, detection: HumanFaceDetection) -> np.ndarray | None:
        if detection.keypoints is None:
            x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_bgr.shape[1], x2)
            y2 = min(image_bgr.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            return cv2.resize(crop, self.recognizer_input_size)

        landmark = detection.keypoints.astype(np.float32)
        ratio = float(self.recognizer_input_size[0]) / 112.0
        dst = ARCFACE_DST.copy() * ratio
        if self.recognizer_input_size[0] % 128 == 0:
            dst[:, 0] += 8.0 * ratio
        matrix, _ = cv2.estimateAffinePartial2D(landmark, dst, method=cv2.LMEDS)
        if matrix is None:
            return None
        return cv2.warpAffine(
            image_bgr,
            matrix,
            self.recognizer_input_size,
            borderValue=0.0,
        )

    def _run_recognizer(self, aligned_faces: list[np.ndarray]) -> list[list[float]]:
        if not aligned_faces:
            return []

        outputs: list[np.ndarray] = []
        if self.recognizer_batchable:
            for start in range(0, len(aligned_faces), self.recognizer_batch_size):
                face_batch = aligned_faces[start : start + self.recognizer_batch_size]
                outputs.append(self._run_recognizer_batch(face_batch))
        else:
            for aligned_face in aligned_faces:
                outputs.append(self._run_recognizer_batch([aligned_face]))

        vectors = np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)
        if vectors.size == 0:
            return [[] for _ in aligned_faces]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normalized = vectors / norms
        return [
            [round(float(value), 6) for value in vector.tolist()]
            for vector in normalized
        ]

    def _run_recognizer_batch(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        blob = cv2.dnn.blobFromImages(
            aligned_faces,
            1.0 / self.recognizer_input_std,
            self.recognizer_input_size,
            (
                self.recognizer_input_mean,
                self.recognizer_input_mean,
                self.recognizer_input_mean,
            ),
            swapRB=True,
        )
        return self.recognizer_session.run(
            [self.recognizer_output_name],
            {self.recognizer_input_name: blob},
        )[0]
