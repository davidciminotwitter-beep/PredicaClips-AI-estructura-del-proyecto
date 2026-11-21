
"""
PredicaClips AI
Módulo de detección y seguimiento facial.

Motor:
- OpenCV YuNet para detectar rostros.
- Asociación temporal para mantener al sujeto principal.
- Selección automática del rostro principal.
- Suavizado de coordenadas.
- Tolerancia ante pérdidas temporales de detección.
- Generación de puntos de seguimiento para smart_crop.py.

Importante:
Este módulo genera coordenadas de seguimiento.
El recorte y el movimiento de cámara se implementan en smart_crop.py.
"""

from __future__ import annotations

import logging
import math

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2

from config import (
    FACE_MODEL_PATH,
    FACE_DETECTION_CONFIDENCE,
    FACE_DETECTION_INTERVAL,
    FACE_MIN_SIZE,
    FACE_LOST_TOLERANCE,
    CAMERA_SMOOTHING_FACTOR,
    FACE_DETECTION_ENABLED,
    FACE_TRACKING_ENABLED,
)


logger = logging.getLogger("PredicaClipsAI.FaceTracker")


# ============================================================
# MODELOS DE DATOS
# ============================================================

@dataclass
class FaceBox:
    x: float
    y: float
    width: float
    height: float
    confidence: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrackingPoint:
    frame: int
    timestamp: float

    center_x: float
    center_y: float

    face_width: float
    face_height: float

    confidence: float

    detected: bool
    subject_found: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# UTILIDADES GEOMÉTRICAS
# ============================================================

def calculate_distance(a: FaceBox, b: FaceBox) -> float:

    dx = a.center_x - b.center_x
    dy = a.center_y - b.center_y

    return math.sqrt(dx * dx + dy * dy)


def calculate_iou(a: FaceBox, b: FaceBox) -> float:

    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)

    x2 = min(
        a.x + a.width,
        b.x + b.width
    )

    y2 = min(
        a.y + a.height,
        b.y + b.height
    )

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    union = a.area + b.area - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def smooth_value(
    previous: float,
    current: float,
    factor: float
) -> float:

    return previous * factor + current * (1 - factor)


# ============================================================
# DETECTOR Y TRACKER
# ============================================================

class FaceTracker:

    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence: float = FACE_DETECTION_CONFIDENCE,
        detection_interval: int = FACE_DETECTION_INTERVAL,
        smoothing_factor: float = CAMERA_SMOOTHING_FACTOR,
    ):

        self.model_path = Path(
            model_path or FACE_MODEL_PATH
        )

        self.confidence = confidence

        self.detection_interval = max(
            1,
            detection_interval
        )

        self.smoothing_factor = max(
            0.0,
            min(0.99, smoothing_factor)
        )

        self.detector = None

        self.previous_face: Optional[FaceBox] = None

        self.last_detection_time: Optional[float] = None

        self.frame_width = 0
        self.frame_height = 0

        self._load_detector()

    # --------------------------------------------------------
    # CARGAR MODELO YUNET
    # --------------------------------------------------------

    def _load_detector(self):

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"No se encontró el modelo facial YuNet: "
                f"{self.model_path}\n"
                "Es necesario instalar el modelo ONNX "
                "antes de utilizar el seguimiento facial."
            )

        if not hasattr(cv2, "FaceDetectorYN_create"):

            raise RuntimeError(
                "Tu versión de OpenCV no incluye FaceDetectorYN. "
                "Instalá una versión compatible de opencv-contrib-python."
            )

        self.detector = cv2.FaceDetectorYN_create(
            str(self.model_path),
            "",
            (320, 320),
            self.confidence,
            0.3,
            5000
        )

        logger.info("Modelo YuNet cargado correctamente.")

    # --------------------------------------------------------
    # DETECTAR TODOS LOS ROSTROS
    # --------------------------------------------------------

    def detect_faces(self, frame) -> list[FaceBox]:

        if self.detector is None:
            return []

        height, width = frame.shape[:2]

        self.detector.setInputSize(
            (width, height)
        )

        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        detected = []

        for face in faces:

            x, y, w, h = face[:4]

            score = float(face[-1])

            if score < self.confidence:
                continue

            if w < FACE_MIN_SIZE or h < FACE_MIN_SIZE:
                continue

            detected.append(
                FaceBox(
                    x=float(x),
                    y=float(y),
                    width=float(w),
                    height=float(h),
                    confidence=score
                )
            )

        return detected

    # --------------------------------------------------------
    # SELECCIONAR EL PREDICADOR PRINCIPAL
    # --------------------------------------------------------

    def select_primary_face(
        self,
        faces: list[FaceBox]
    ) -> Optional[FaceBox]:

        if not faces:
            return None

        # Si ya tenemos una persona identificada,
        # intentamos conservar su identidad.

        if self.previous_face is not None:

            candidates = []

            for face in faces:

                distance = calculate_distance(
                    self.previous_face,
                    face
                )

                iou = calculate_iou(
                    self.previous_face,
                    face
                )

                # Distancia máxima permitida:
                # depende del tamaño de la imagen.

                max_distance = max(
                    self.frame_width * 0.25,
                    self.previous_face.width * 2.5
                )

                if distance <= max_distance or iou > 0.05:

                    score = (
                        (iou * 3.0)
                        + (face.area / max(
                            self.frame_width * self.frame_height,
                            1
                        ))
                        - (distance / max(
                            self.frame_width,
                            1
                        ))
                    )

                    candidates.append(
                        (score, face)
                    )

            if candidates:

                candidates.sort(
                    key=lambda item: item[0],
                    reverse=True
                )

                return candidates[0][1]

        # Primera detección:
        # priorizamos el rostro más grande.
        # Es una heurística, no una identificación garantizada.

        return max(
            faces,
            key=lambda face: face.area
        )

    # --------------------------------------------------------
    # SUAVIZAR EL MOVIMIENTO
    # --------------------------------------------------------

    def smooth_face(
        self,
        face: FaceBox
    ) -> FaceBox:

        if self.previous_face is None:
            return face

        factor = self.smoothing_factor

        return FaceBox(
            x=smooth_value(
                self.previous_face.x,
                face.x,
                factor
            ),

            y=smooth_value(
                self.previous_face.y,
                face.y,
                factor
            ),

            width=smooth_value(
                self.previous_face.width,
                face.width,
                factor
            ),

            height=smooth_value(
                self.previous_face.height,
                face.height,
                factor
            ),

            confidence=face.confidence
        )

    # --------------------------------------------------------
    # ANALIZAR VIDEO COMPLETO
    # --------------------------------------------------------

    def analyze_video(
        self,
        video_path: str | Path
    ) -> list[TrackingPoint]:

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"No existe el video: {video_path}"
            )

        if not FACE_DETECTION_ENABLED:
            raise RuntimeError(
                "La detección facial está desactivada."
            )

        if not FACE_TRACKING_ENABLED:
            raise RuntimeError(
                "El seguimiento facial está desactivado."
            )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():

            raise RuntimeError(
                f"No se pudo abrir el video: {video_path}"
            )

        fps = capture.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30.0

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.frame_width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.frame_height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.previous_face = None
        self.last_detection_time = None

        tracking_data = []

        frame_number = 0

        logger.info(
            "Iniciando seguimiento facial: %s",
            video_path.name
        )

        try:

            while True:

                success, frame = capture.read()

                if not success:
                    break

                timestamp = frame_number / fps

                should_detect = (
                    frame_number % self.detection_interval == 0
                    or self.previous_face is None
                )

                detected = False
                selected_face = None

                if should_detect:

                    faces = self.detect_faces(frame)

                    selected_face = self.select_primary_face(
                        faces
                    )

                    if selected_face is not None:

                        selected_face = self.smooth_face(
                            selected_face
                        )

                        self.previous_face = selected_face

                        self.last_detection_time = timestamp

                        detected = True

                # Si no hubo detección en este fotograma,
                # conservamos temporalmente la última posición.

                if self.previous_face is not None:

                    elapsed_since_detection = (
                        timestamp - self.last_detection_time
                        if self.last_detection_time is not None
                        else float("inf")
                    )

                    subject_found = (
                        elapsed_since_detection
                        <= FACE_LOST_TOLERANCE
                    )

                else:

                    subject_found = False

                if subject_found:

                    face = self.previous_face

                    point = TrackingPoint(
                        frame=frame_number,
                        timestamp=timestamp,

                        center_x=face.center_x,
                        center_y=face.center_y,

                        face_width=face.width,
                        face_height=face.height,

                        confidence=face.confidence,

                        detected=detected,
                        subject_found=True
                    )

                else:

                    # No inventamos una posición facial.
                    # El siguiente módulo podrá aplicar
                    # el encuadre alternativo.

                    point = TrackingPoint(
                        frame=frame_number,
                        timestamp=timestamp,

                        center_x=self.frame_width / 2,
                        center_y=self.frame_height / 2,

                        face_width=0,
                        face_height=0,

                        confidence=0,

                        detected=False,
                        subject_found=False
                    )

                    if self.previous_face is not None:
                        self.previous_face = None
                        self.last_detection_time = None

                tracking_data.append(point)

                frame_number += 1

                if frame_number % 500 == 0:

                    logger.info(
                        "Procesados %s / %s fotogramas",
                        frame_number,
                        total_frames
                    )

        finally:

            capture.release()

        logger.info(
            "Seguimiento finalizado. Fotogramas: %s",
            len(tracking_data)
        )

        return tracking_data

    # --------------------------------------------------------
    # EXPORTAR RESULTADOS
    # --------------------------------------------------------

    @staticmethod
    def export_tracking(
        tracking_data: list[TrackingPoint],
        output_path: str | Path
    ):

        import json

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = [
            point.to_dict()
            for point in tracking_data
        ]

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

        logger.info(
            "Datos de seguimiento guardados: %s",
            output_path
        )
