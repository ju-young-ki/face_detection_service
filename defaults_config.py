"""defaults.json 기본 설정 로더.

요청에 파라미터가 없으면(또는 None이면) JSON에 저장된 기본값을 사용한다.
설정은 process / passport / crop 구분 없이 하나의 객체로 관리한다.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from paths import app_dir, resource_dir


def _resolve_config_path() -> Path:
    """실행 파일 옆 defaults.json을 우선하고, 없으면 번들 경로를 사용한다."""
    beside = app_dir() / "defaults.json"
    if beside.is_file():
        return beside
    bundled = resource_dir() / "defaults.json"
    if bundled.is_file():
        return bundled
    return beside


_CONFIG_PATH = _resolve_config_path()

# JSON 파일이 없거나 깨졌을 때 사용할 내장 기본값
_BUILTIN: dict[str, Any] = {
    "whitening": 0.45,
    "smooth": 0.55,
    "sharpness": 0.0,
    "gamma": 1.0,
    "contrast": 0.0,
    "red": 0.0,
    "green": 0.0,
    "blue": 0.0,
    "temperature": 0.0,
    "hue": 0.0,
    "saturation": 0.0,
    "cutout": False,
    "crop_category": "420",
    "face_size": 10,
    "top_margin_mm": 2.0,
}

_cached: dict[str, Any] | None = None


def config_path() -> Path:
    return _CONFIG_PATH


def _flatten_if_legacy(raw: dict[str, Any]) -> dict[str, Any]:
    """옛 process/passport/crop 섹션 형식이면 평탄화한다."""
    legacy_keys = ("process", "passport", "crop")
    if not any(isinstance(raw.get(k), dict) for k in legacy_keys):
        return raw

    flat: dict[str, Any] = {}
    for key in legacy_keys:
        section = raw.get(key)
        if isinstance(section, dict):
            flat.update(section)
    for key, value in raw.items():
        if key not in legacy_keys:
            flat[key] = value
    return flat


def load_defaults(*, force_reload: bool = False) -> dict[str, Any]:
    """defaults.json을 읽어 통합 기본값을 반환한다."""
    global _cached
    if _cached is not None and not force_reload:
        return deepcopy(_cached)

    merged = deepcopy(_BUILTIN)
    if _CONFIG_PATH.is_file():
        try:
            with _CONFIG_PATH.open(encoding="utf-8") as fp:
                raw = json.load(fp)
            if isinstance(raw, dict):
                flat = _flatten_if_legacy(raw)
                merged.update({k: v for k, v in flat.items() if not isinstance(v, dict)})
        except (OSError, json.JSONDecodeError):
            pass

    _cached = merged
    return deepcopy(_cached)


def get_defaults() -> dict[str, Any]:
    """통합 기본값 복사본을 반환한다."""
    return load_defaults()


def resolve_params(**provided: Any) -> dict[str, Any]:
    """제공된 값이 None이면 JSON 기본값으로 채운다.

    예:
        resolve_params(whitening=None, smooth=0.7)
        → whitening은 defaults.json, smooth는 0.7
    """
    result = load_defaults()
    for key, value in provided.items():
        if value is not None:
            result[key] = value
    return result
