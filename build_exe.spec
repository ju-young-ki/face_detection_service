# -*- mode: python ; coding: utf-8 -*-
"""증명사진 미백 · 누끼 데스크톱 앱 PyInstaller 스펙."""

from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

ctk_datas, ctk_binaries, ctk_hidden = collect_all("customtkinter")
mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=ctk_binaries + mp_binaries,
    datas=(
        ctk_datas
        + mp_datas
        + collect_data_files("customtkinter")
        + [("defaults.json", ".")]
        + [("models/face_landmarker.task", "models")]
        + [("models/selfie_segmenter.tflite", "models")]
    ),
    hiddenimports=ctk_hidden + mp_hidden + [
        "PIL._tkinter_finder",
        "defaults_config",
        "processor",
        "paths",
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
    name="증명사진미백",
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
    name="증명사진미백",
)
