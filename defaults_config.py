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
    """실행 파일 옆 defaults.json을 우선하고, 없으면 번들 경로를 사용한다.

    매번 호출할 때마다 파일 존재 여부를 재확인하므로
    실행 중에 파일이 생성되거나 위치가 바뀌어도 올바른 경로를 반환한다.
    """
    beside = app_dir() / "defaults.json"
    if beside.is_file():
        return beside
    bundled = resource_dir() / "defaults.json"
    if bundled.is_file():
        return bundled
    return beside


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
_cache_mtime: float = 0.0


def config_path() -> Path:
    return _resolve_config_path()


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
    """defaults.json을 읽어 통합 기본값을 반환한다.

    force_reload=False 이어도 실행 파일 옆 defaults.json이
    캐시 이후 변경되었으면 자동으로 다시 읽는다.
    """
    global _cached, _cache_mtime

    config = _resolve_config_path()

    if _cached is not None and not force_reload:
        # 파일이 존재하고, 캐시 후 변경된 경우 자동 무효화
        try:
            if not config.is_file() or config.stat().st_mtime <= _cache_mtime:
                return deepcopy(_cached)
        except OSError:
            return deepcopy(_cached)

    merged = deepcopy(_BUILTIN)
    if config.is_file():
        try:
            with config.open(encoding="utf-8") as fp:
                raw = json.load(fp)
            if isinstance(raw, dict):
                flat = _flatten_if_legacy(raw)
                merged.update({k: v for k, v in flat.items() if not isinstance(v, dict)})
        except (OSError, json.JSONDecodeError):
            pass

    _cached = merged
    try:
        _cache_mtime = config.stat().st_mtime if config.is_file() else 0.0
    except OSError:
        _cache_mtime = 0.0
    return deepcopy(_cached)


def get_defaults() -> dict[str, Any]:
    """통합 기본값 복사본을 반환한다."""
    return load_defaults()


def _json_safe(value: Any) -> Any:
    """JSON에 쓰기 좋게 값을 정규화한다."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return round(float(value), 4)
    if isinstance(value, str):
        return value
    return value


def save_defaults(updates: dict[str, Any] | None = None, **kwargs: Any) -> Path:
    """현재 설정을 실행 파일(또는 프로젝트) 옆 defaults.json에 저장한다.

    UI에 없는 키(crop_category, face_size 등)는 기존 값을 유지한다.
    저장 후 캐시를 즉시 갱신하므로 서버가 재시작 없이 새 값을 사용한다.
    """
    global _cached, _cache_mtime

    current = load_defaults(force_reload=True)
    if updates:
        current.update(updates)
    current.update(kwargs)

    payload = {key: _json_safe(current.get(key, _BUILTIN[key])) for key in _BUILTIN}

    target = app_dir() / "defaults.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")

    _cached = deepcopy(payload)
    try:
        _cache_mtime = target.stat().st_mtime
    except OSError:
        _cache_mtime = 0.0
    return target


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
