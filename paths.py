"""실행 환경(소스 / PyInstaller)에 맞는 리소스 경로."""

from __future__ import annotations

import sys
from pathlib import Path


def resource_dir() -> Path:
    """번들된 리소스(모델 등)가 있는 디렉터리."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


def app_dir() -> Path:
    """실행 파일(또는 프로젝트) 옆 디렉터리. 사용자 편집용."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent
