
"""
PredicaClips AI
Configuración central de la aplicación.

Responsabilidades:
- Rutas y directorios.
- Gemini y análisis inteligente.
- Transcripción Whisper.
- Detección y seguimiento facial.
- Encuadre vertical inteligente.
- Subtítulos dinámicos.
- Procesamiento de audio.
- Render cinematográfico.
- Miniaturas y exportación.

Python 3.10+
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ============================================================
# 1. RUTAS PRINCIPALES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODULES_DIR = BASE_DIR / "modules"
ASSETS_DIR = BASE_DIR / "assets"

MUSIC_DIR = ASSETS_DIR / "music"
FONTS_DIR = ASSETS_DIR / "fonts"
SOUNDS_DIR = ASSETS_DIR / "sounds"
MODELS_DIR = ASSETS_DIR / "models"

TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

for directory in (
    MODULES_DIR,
    MUSIC_DIR,
    FONTS_DIR,
    SOUNDS_DIR,
    MODELS_DIR,
    TEMP_DIR,
    OUTPUT_DIR,
    LOGS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. VARIABLES DE ENTORNO
# ============================================================

def load_env_file() -> None:
    """
    Carga variables desde .env sin sobrescribir
    variables definidas en el sistema operativo.
    """

    env_path = BASE_DIR / ".env"

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)

        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            os.environ.setdefault(key, value)


load_env_file()


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1", "true", "yes", "si", "sí", "on"
    }


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ============================================================
# 3. CONFIGURACIÓN GENERAL
# ============================================================

APP_NAME = "PredicaClips AI"
APP_VERSION = "1.0.0"
APP_ENV = os.getenv("APP_ENV", "production")

DEBUG = env_bool("DEBUG", False)

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "DEBUG" if DEBUG else "INFO"
).upper()


# ============================================================
# 4. INTELIGENCIA ARTIFICIAL - GEMINI
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

GEMINI_TEMPERATURE = 0.35

GEMINI_MAX_OUTPUT_TOKENS = 8192

GEMINI_TIMEOUT_SECONDS = 120

GEMINI_RETRIES = 3

# La aplicación debe analizar el contexto completo
# de la predicación, no solamente frases aisladas.

ANALYSIS_SETTINGS = {
    "target_clips": 4,
    "min_duration_seconds": 20,
    "max_duration_seconds": 90,

    "evaluate": [
        "impacto_emocional",
        "claridad_del_mensaje",
        "fuerza_del_inicio",
        "interes_para_la_audiencia",
        "autonomia_del_fragmento",
        "potencial_de_compartirse",
        "coherencia_biblica",
    ],

    "require_biblical_context": True,
    "avoid_out_of_context_quotes": True,
    "avoid_repeating_same_idea": True,
    "require_complete_sentences": True,
}


# ============================================================
# 5. TRANSCRIPCIÓN - FASTER WHISPER
# ============================================================

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "small"
)

WHISPER_DEVICE = os.getenv(
    "WHISPER_DEVICE",
    "auto"
)

WHISPER_COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "default"
)

WHISPER_LANGUAGE = "es"

WHISPER_BEAM_SIZE = 5

WHISPER_VAD_FILTER = True

WHISPER_WORD_TIMESTAMPS = True

WHISPER_CONDITION_ON_PREVIOUS_TEXT = True

WHISPER_INITIAL_PROMPT = (
    "Predicación cristiana evangélica en español. "
    "Terminología bíblica, nombres de libros de la Biblia, "
    "versículos, oración, fe, Jesucristo y Espíritu Santo."
)

TRANSCRIPTION_BATCH_SIZE = 16


# ============================================================
# 6. DETECCIÓN Y SEGUIMIENTO DE ROSTROS
# ============================================================

FACE_DETECTION_ENABLED = True

FACE_TRACKING_ENABLED = True

# Detector configurable para no atar el proyecto
# a un único modelo de detección.

FACE_DETECTOR = os.getenv(
    "FACE_DETECTOR",
    "opencv_yunet"
)

FACE_MODEL_PATH = (
    MODELS_DIR / "face_detection_yunet_2023mar.onnx"
)

FACE_DETECTION_CONFIDENCE = 0.75

FACE_MIN_SIZE = 40

FACE_DETECTION_INTERVAL = 5

FACE_TRACKING_METHOD = "CSRT"

FACE_REDETECT_INTERVAL = 30

FACE_LOST_TOLERANCE = 2.0

# Cuando aparecen varias personas, se intenta mantener
# la identidad del predicador seleccionado.

FACE_IDENTITY_TRACKING = True

FACE_PRIMARY_SUBJECT_PRIORITY = [
    "selected_subject",
    "largest_face",
    "center_face",
    "persistent_track",
]

# Espacio adicional alrededor del rostro.

FACE_PADDING_HORIZONTAL = 0.35
FACE_PADDING_VERTICAL = 0.65

# Suavizado de movimientos de encuadre.

CAMERA_SMOOTHING_ENABLED = True

CAMERA_SMOOTHING_FACTOR = 0.82

CAMERA_MAX_MOVEMENT_PER_SECOND = 0.75

CAMERA_REACQUIRE_ENABLED = True


# ============================================================
# 7. ENCUADRE INTELIGENTE - VERTICAL 9:16
# ============================================================

SMART_CROP_ENABLED = True

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

OUTPUT_ASPECT_RATIO = 9 / 16

OUTPUT_FPS = 30

CROP_FOCUS_PRIORITY = [
    "face",
    "upper_body",
    "person",
    "original_center",
]

CROP_KEEP_HEAD_VISIBLE = True

CROP_KEEP_SHOULDERS_VISIBLE = True

CROP_ALLOW_HORIZONTAL_MOVEMENT = True

CROP_SMOOTH_TRANSITIONS = True

CROP_FALLBACK_TO_CENTER = True

# Evita acercamientos exagerados.

CROP_MIN_ZOOM = 1.0
CROP_MAX_ZOOM = 2.2

CROP_ZOOM_SMOOTHING = 0.90


# ============================================================
# 8. SUBTÍTULOS DINÁMICOS
# ============================================================

SUBTITLES_ENABLED = True

SUBTITLE_FORMAT = "ass"

SUBTITLE_FONT = "Montserrat ExtraBold"

SUBTITLE_FONT_SIZE = 78

SUBTITLE_MAX_WORDS_PER_GROUP = 3

SUBTITLE_MAX_LINES = 2

SUBTITLE_POSITION = "lower_center"

SUBTITLE_VERTICAL_MARGIN = 330

SUBTITLE_OUTLINE = 5

SUBTITLE_SHADOW = 2

SUBTITLE_BOLD = True

SUBTITLE_WORD_HIGHLIGHT = True

SUBTITLE_ANIMATION_ENABLED = True

SUBTITLE_FADE_IN_MS = 80

SUBTITLE_FADE_OUT_MS = 80

# Colores ASS: formato &HAABBGGRR.

SUBTITLE_COLORS = {
    "normal": "&H00FFFFFF",       # Blanco
    "highlight": "&H0000FFFF",    # Amarillo
    "emphasis": "&H000000FF",     # Rojo
}

SUBTITLE_OUTLINE_COLOR = "&H00000000"

SUBTITLE_BACKGROUND_COLOR = "&H80000000"

SUBTITLE_SAFE_MARGIN = 80


# ============================================================
# 9. AUDIO Y MÚSICA CINEMATOGRÁFICA
# ============================================================

AUDIO_ENABLED = True

VOICE_VOLUME = 1.0

MUSIC_ENABLED = True

MUSIC_VOLUME = 0.12

MUSIC_VOLUME_MIN = 0.04

MUSIC_VOLUME_MAX = 0.18

AUDIO_DUCKING_ENABLED = True

AUDIO_DUCKING_THRESHOLD = 0.025

AUDIO_DUCKING_RATIO = 8

AUDIO_DUCKING_ATTACK_MS = 150

AUDIO_DUCKING_RELEASE_MS = 900

AUDIO_FADE_IN_SECONDS = 2.0

AUDIO_FADE_OUT_SECONDS = 3.0

AUDIO_NORMALIZE_VOICE = True

AUDIO_TARGET_LUFS = -14

AUDIO_TRUE_PEAK_DB = -1.0

AUDIO_SAMPLE_RATE = 48000

AUDIO_CHANNELS = 2

# La selección musical se realizará según el contenido
# emocional y el ritmo de cada fragmento.

MUSIC_MOOD_CATEGORIES = [
    "inspirational",
    "epic",
    "emotional",
    "dramatic",
    "hopeful",
    "reflective",
]

MUSIC_AUTO_SELECTION = True

MUSIC_REQUIRE_COMMERCIAL_LICENSE = True

MUSIC_STORE_LICENSE_REFERENCE = True


# ============================================================
# 10. EDICIÓN CINEMATOGRÁFICA
# ============================================================

CINEMATIC_MODE = True

COLOR_CORRECTION_ENABLED = True

CONTRAST = 1.05

SATURATION = 1.08

SHARPNESS = 1.05

BRIGHTNESS = 0.0

ZOOM_EFFECTS_ENABLED = True

ZOOM_EFFECT_INTENSITY = 1.08

ZOOM_TRANSITIONS_ENABLED = True

TRANSITIONS_ENABLED = False

# Las transiciones se mantienen desactivadas inicialmente
# para evitar cortes artificiales dentro de una predicación.

STABILIZATION_ENABLED = False


# ============================================================
# 11. MINIATURAS Y PORTADAS
# ============================================================

THUMBNAILS_ENABLED = True

THUMBNAIL_WIDTH = 1080

THUMBNAIL_HEIGHT = 1920

THUMBNAIL_FACE_PRIORITY = True

THUMBNAIL_MIN_FACE_CONFIDENCE = 0.75

THUMBNAIL_AVOID_CLOSED_EYES = True

THUMBNAIL_AVOID_BLURRED_FRAMES = True

THUMBNAIL_TEXT_MAX_WORDS = 7

THUMBNAIL_FONT = "Montserrat ExtraBold"

THUMBNAIL_ALLOW_CUSTOM_PROMPT = True

THUMBNAIL_USE_VIDEO_FRAMES = True


# ============================================================
# 12. EXPORTACIÓN DE VIDEO
# ============================================================

VIDEO_FORMAT = "mp4"

VIDEO_CODEC = "libx264"

VIDEO_PRESET = os.getenv(
    "VIDEO_PRESET",
    "medium"
)

VIDEO_CRF = 18

VIDEO_PIXEL_FORMAT = "yuv420p"

VIDEO_AUDIO_CODEC = "aac"

VIDEO_AUDIO_BITRATE = "192k"

VIDEO_FASTSTART = True

VIDEO_THREADS = 0

VIDEO_OUTPUT_SUFFIX = "_predicaclip"


# ============================================================
# 13. LÍMITES DE PROCESAMIENTO
# ============================================================

TARGET_CLIPS = 4

MIN_CLIP_DURATION = 20

MAX_CLIP_DURATION = 90

MAX_VIDEO_SIZE_MB = 4096

MAX_VIDEO_DURATION_MINUTES = 180

DELETE_TEMP_FILES_AFTER_RENDER = True

KEEP_TRANSCRIPTION = True

KEEP_ANALYSIS_JSON = True


# ============================================================
# 14. HERRAMIENTAS EXTERNAS
# ============================================================

FFMPEG_PATH = os.getenv(
    "FFMPEG_PATH",
    shutil.which("ffmpeg") or "ffmpeg"
)

FFPROBE_PATH = os.getenv(
    "FFPROBE_PATH",
    shutil.which("ffprobe") or "ffprobe"
)


# ============================================================
# 15. REGISTRO DE ACTIVIDAD
# ============================================================

LOG_FILE = LOGS_DIR / "predicaclips.log"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(APP_NAME)


# ============================================================
# 16. VALIDACIÓN GENERAL
# ============================================================

@dataclass(frozen=True)
class ConfigurationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0


def validate_configuration() -> ConfigurationReport:

    errors = []
    warnings = []

    if TARGET_CLIPS < 1:
        errors.append("TARGET_CLIPS debe ser mayor que cero.")

    if MIN_CLIP_DURATION <= 0:
        errors.append("La duración mínima debe ser positiva.")

    if MIN_CLIP_DURATION >= MAX_CLIP_DURATION:
        errors.append(
            "La duración mínima debe ser menor que la máxima."
        )

    if OUTPUT_WIDTH <= 0 or OUTPUT_HEIGHT <= 0:
        errors.append("La resolución de salida no es válida.")

    if MUSIC_VOLUME < 0 or MUSIC_VOLUME > 1:
        errors.append("MUSIC_VOLUME debe estar entre 0 y 1.")

    if not 0 < CAMERA_SMOOTHING_FACTOR <= 1:
        errors.append(
            "CAMERA_SMOOTHING_FACTOR debe estar entre 0 y 1."
        )

    if not 0 < FACE_DETECTION_CONFIDENCE <= 1:
        errors.append(
            "FACE_DETECTION_CONFIDENCE debe estar entre 0 y 1."
        )

    if not GEMINI_API_KEY:
        warnings.append(
            "No se configuró GEMINI_API_KEY. "
            "El análisis inteligente no estará disponible."
        )

    if not shutil.which(FFMPEG_PATH) and not Path(FFMPEG_PATH).exists():
        warnings.append(
            "FFmpeg no fue encontrado. "
            "No se podrá exportar hasta configurarlo."
        )

    if not FACE_MODEL_PATH.exists():
        warnings.append(
            "No se encontró el modelo YuNet. "
            "La detección facial necesita descargarlo."
        )

    if not MUSIC_DIR.exists():
        warnings.append("No existe la carpeta de música.")

    return ConfigurationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def print_configuration_report() -> ConfigurationReport:

    report = validate_configuration()

    logger.info("Validando configuración de %s", APP_NAME)

    for warning in report.warnings:
        logger.warning(warning)

    for error in report.errors:
        logger.error(error)

    if report.valid:
        logger.info("Configuración básica validada correctamente.")

    return report


# ============================================================
# 17. CONFIGURACIÓN INICIAL
# ============================================================

CONFIG_REPORT = print_configuration_report()

if not CONFIG_REPORT.valid:
    raise RuntimeError(
        "La configuración de PredicaClips AI contiene errores: "
        + "; ".join(CONFIG_REPORT.errors)
    )
