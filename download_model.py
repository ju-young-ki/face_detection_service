"""Download MediaPipe models if missing."""

from __future__ import annotations

import urllib.request
from pathlib import Path

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
SEGMENTER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
SEGMENTER_MODEL_PATH = MODELS_DIR / "selfie_segmenter.tflite"


def _download(url: str, path: Path, label: str) -> Path:
    if path.is_file():
        return path

    print(f"{label} 다운로드 중...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    print(f"저장됨: {path}")
    return path


def ensure_models() -> tuple[Path, Path]:
    face = _download(FACE_MODEL_URL, FACE_MODEL_PATH, "얼굴 인식 모델")
    segmenter = _download(SEGMENTER_MODEL_URL, SEGMENTER_MODEL_PATH, "누끼 모델")
    return face, segmenter


def ensure_model() -> Path:
    face, _ = ensure_models()
    return face


if __name__ == "__main__":
    ensure_models()
