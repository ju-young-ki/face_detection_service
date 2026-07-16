# -*- mode: python ; coding: utf-8 -*-
"""Face Detection FastAPI 서버 PyInstaller 스펙."""

from PyInstaller.utils.hooks import collect_all

block_cipher = None

mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")
uvicorn_datas, uvicorn_binaries, uvicorn_hidden = collect_all("uvicorn")
fastapi_datas, fastapi_binaries, fastapi_hidden = collect_all("fastapi")

a = Analysis(
    ["server.py"],
    pathex=[],
    binaries=mp_binaries + uvicorn_binaries + fastapi_binaries,
    datas=(
        mp_datas
        + uvicorn_datas
        + fastapi_datas
        + [("defaults.json", ".")]
        + [("models/face_landmarker.task", "models")]
        + [("models/selfie_segmenter.tflite", "models")]
    ),
    hiddenimports=mp_hidden
    + uvicorn_hidden
    + fastapi_hidden
    + [
        "defaults_config",
        "processor",
        "paths",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "multipart",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FaceDetectionServer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FaceDetectionServer",
)
