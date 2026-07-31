"""증명사진 미백·누끼 - FastAPI 로컬 서버.

실행 방법
---------
    # 가상환경 활성화 후
    python server.py
    # 또는
    uvicorn server:app --host 0.0.0.0 --port 5000

기본 주소: http://127.0.0.1:5000
Swagger UI: http://127.0.0.1:5000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from defaults_config import load_defaults, resolve_params
from processor import PhotoProcessor

DEFAULT_PORT = 5000

# ---------------------------------------------------------------------------
# 전역 처리기
# MediaPipe 모델은 로딩 비용이 크므로, 서버 기동 시 한 번만 생성하고 재사용한다.
# 처리 파라미터 기본값은 defaults.json 에서 로드한다.
# ---------------------------------------------------------------------------
_processor: PhotoProcessor | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """앱 시작/종료 생명주기.

    - 시작: defaults.json 로드 + PhotoProcessor(미백·누끼·얼굴감지) 초기화
    - 종료: 모델 리소스 해제
    """
    global _processor
    load_defaults(force_reload=True)
    _processor = PhotoProcessor()
    yield
    if _processor is not None:
        _processor.close()
        _processor = None


app = FastAPI(
    title="Face Detection Service",
    description="증명사진 미백·누끼·크롭 이미지 처리 API",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_processor() -> PhotoProcessor:
    """초기화가 끝난 PhotoProcessor를 반환한다. 준비 전이면 503."""
    if _processor is None:
        raise HTTPException(status_code=503, detail="처리기가 아직 준비되지 않았습니다.")
    return _processor


def _process_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """processor.process()에 넘길 키만 추린다."""
    keys = (
        "whitening",
        "smooth",
        "sharpness",
        "gamma",
        "contrast",
        "red",
        "green",
        "blue",
        "temperature",
        "hue",
        "saturation",
        "forehead_shine",
        "cutout",
    )
    return {k: params[k] for k in keys}


async def _read_image_bgr(file: UploadFile) -> np.ndarray:
    """업로드 파일을 OpenCV BGR ndarray로 디코딩한다.

    Content-Type이 없거나 application/octet-stream 이어도
    실제 바이트로 이미지 디코딩을 시도한다. (Flutter 등 클라이언트 대응)
    """
    content_type = (file.content_type or "").lower()
    if content_type and not (
        content_type.startswith("image/")
        or content_type in ("application/octet-stream", "binary/octet-stream")
    ):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다.")
    return image


def _encode_image(image: np.ndarray, *, cutout: bool = False) -> tuple[bytes, str]:
    """처리 결과를 HTTP 응답용 바이트로 인코딩한다.

    - cutout=True 이고 알파 채널(BGRA)이 있으면 → PNG (투명 배경 유지)
    - 그 외 → 흰 배경 JPEG (품질 95)
    """
    if cutout and image.ndim == 3 and image.shape[2] == 4:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise HTTPException(status_code=500, detail="PNG 인코딩에 실패했습니다.")
        return encoded.tobytes(), "image/png"

    output = PhotoProcessor.composite_on_white(image)
    ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG 인코딩에 실패했습니다.")
    return encoded.tobytes(), "image/jpeg"


def _image_response(body: bytes, media_type: str, filename: str) -> Response:
    """다운로드/저장이 쉽도록 Content-Disposition 을 포함한 이미지 응답."""
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


# ===========================================================================
# 유틸 엔드포인트
# ===========================================================================


@app.get("/")
async def root() -> dict[str, str]:
    """서비스 안내. docs·health·defaults·crop URL을 돌려준다."""
    return {
        "service": "face-detection-service",
        "docs": "/docs",
        "health": "/health",
        "defaults": "/api/defaults",
        "crop": "/crop",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """헬스체크. 로드밸런서·모니터링용으로 status=ok 를 반환한다."""
    return {"status": "ok"}


@app.get("/api/defaults")
async def api_defaults() -> dict[str, Any]:
    """defaults.json에 저장된 기본 설정값을 반환한다."""
    return load_defaults()


# ===========================================================================
# /crop — 미백·누끼 적용 후 crop_category 규격으로 얼굴 기준 크롭
#
# Content-Type: multipart/form-data
#   image         : 이미지 파일 (필수)
#   crop_category : 크롭 종류 (선택, 기본값 defaults.json)
#                   420 → 420×540 (3.5×4.5)
#                   360 → 360×480 (3×4)
#                   600 → 600×600 (5×5)
#                   720 → 720×840 (6×7)
#                   540 → 540×660 (4.5×5.5)
#   face_size     : 얼굴 크기 (정수, 기준 10, 범위 1~30)
#   top_margin_mm : 머리 위 여백(mm) (선택)
#   cutout        : 배경제거 (선택, 기본값 defaults.json — 보통 true)
#   + whitening/smooth 등 보정 파라미터 (생략 시 defaults.json)
#
# 응답: image/jpeg (흰 배경·규격 크기)
# 실패: 얼굴 미검출·잘못된 category 시 400
# ===========================================================================


@app.post("/crop")
async def crop_image(
    image: Annotated[UploadFile, File(description="크롭할 이미지")],
    crop_category: Annotated[str | None, Form(description="크롭 종류")] = None,
    face_size: Annotated[int | None, Form(description="얼굴 크기 (기준 10)")] = None,
    top_margin_mm: Annotated[float | None, Form(description="머리 위 여백(mm)")] = None,
    cutout: Annotated[bool | None, Form(description="배경 제거(누끼)")] = None,
    whitening: Annotated[float | None, Form()] = None,
    smooth: Annotated[float | None, Form()] = None,
    sharpness: Annotated[float | None, Form()] = None,
    gamma: Annotated[float | None, Form()] = None,
    contrast: Annotated[float | None, Form()] = None,
    red: Annotated[float | None, Form()] = None,
    green: Annotated[float | None, Form()] = None,
    blue: Annotated[float | None, Form()] = None,
    temperature: Annotated[float | None, Form()] = None,
    hue: Annotated[float | None, Form()] = None,
    saturation: Annotated[float | None, Form()] = None,
    forehead_shine: Annotated[float | None, Form(description="이마 광택 제거(0~1)")] = None,
) -> Response:
    """미백·누끼 등을 적용한 뒤 crop_category 규격으로 크롭한 JPEG를 반환한다."""
    params = resolve_params(
        crop_category=crop_category,
        face_size=face_size,
        top_margin_mm=top_margin_mm,
        cutout=cutout,
        whitening=whitening,
        smooth=smooth,
        sharpness=sharpness,
        gamma=gamma,
        contrast=contrast,
        red=red,
        green=green,
        blue=blue,
        temperature=temperature,
        hue=hue,
        saturation=saturation,
        forehead_shine=forehead_shine,
    )
    category = str(params["crop_category"])
    size = int(params["face_size"])
    margin = float(params["top_margin_mm"])

    source = await _read_image_bgr(image)
    processor = _get_processor()

    try:
        processed = processor.process(source, **_process_kwargs(params))
        cropped = processor.crop_by_category(
            processed,
            category,
            top_margin_mm=margin,
            face_size=size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ok, encoded = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG 인코딩에 실패했습니다.")

    return _image_response(
        encoded.tobytes(),
        "image/jpeg",
        f"crop_{category}.jpg",
    )


# ===========================================================================
# /api/face-detect/preview — 얼굴 박스 그린 미리보기 이미지
#
# Content-Type: multipart/form-data
#   file : 이미지 파일 (필수)
#
# 응답: image/jpeg (얼굴 영역에 사각형이 그려진 이미지)
# ===========================================================================


@app.post("/api/face-detect/preview")
async def face_detect_preview(
    file: Annotated[UploadFile, File(description="얼굴 표시 미리보기 이미지")],
) -> Response:
    """얼굴 영역에 사각형을 그린 이미지를 반환한다."""
    source = await _read_image_bgr(file)
    processor = _get_processor()
    annotated = processor.draw_face_boxes(source)
    body, media_type = _encode_image(annotated)
    ext = "png" if media_type == "image/png" else "jpg"
    return _image_response(body, media_type, f"faces_preview.{ext}")


def _ensure_port_available(host: str, port: int) -> None:
    """포트가 이미 사용 중이면 원인 안내 후 종료한다."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host if host != "0.0.0.0" else "127.0.0.1", port))
        except OSError as exc:
            raise SystemExit(
                f"포트 {port} 를 사용할 수 없습니다 ({exc}).\n"
                "다른 프로그램이 이미 점유 중인지 확인해주세요."
            ) from exc


if __name__ == "__main__":
    import multiprocessing
    import sys
    from pathlib import Path

    multiprocessing.freeze_support()

    # pythonw / PyInstaller --noconsole 에서는 stdout·stderr 가 None 이라
    # 로깅·print 시 바로 종료될 수 있다. 실행 파일 옆 server.log 로 보낸다.
    if getattr(sys, "frozen", False) and (sys.stdout is None or sys.stderr is None):
        log_path = Path(sys.executable).resolve().parent / "server.log"
        log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = log_fp
        if sys.stderr is None:
            sys.stderr = log_fp

    import uvicorn

    host = "0.0.0.0"
    port = DEFAULT_PORT
    _ensure_port_available(host, port)
    print(f"Face Detection Server: http://127.0.0.1:{port}/docs")

    # PyInstaller 실행 시 모듈 문자열("server:app") 대신 앱 객체를 직접 전달
    uvicorn.run(app, host=host, port=port, reload=False)
