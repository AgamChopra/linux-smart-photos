# -*- mode: python ; coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


ROOT = Path(SPECPATH).resolve().parents[1]
ENTRYPOINT = ROOT / "src" / "linux_smart_photos" / "launcher.py"
ICON_FILE = ROOT / "assets" / "smart-photos.svg"

datas = [(str(ICON_FILE), "assets")]
binaries = []
hiddenimports: list[str] = []


def optional_collect_all(module_name: str) -> None:
    global datas, binaries, hiddenimports
    try:
        module_datas, module_binaries, module_hiddenimports = collect_all(module_name)
    except Exception:
        return
    datas += module_datas
    binaries += module_binaries
    hiddenimports += module_hiddenimports


def optional_copy_metadata(distribution_name: str) -> None:
    global datas
    try:
        datas += copy_metadata(distribution_name)
    except Exception:
        return


for module_name in (
    "huggingface_hub",
    "insightface",
    "onnxruntime",
    "pillow_heif",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
    "ultralytics",
):
    optional_collect_all(module_name)

for distribution_name in (
    "linux-smart-photos",
    "huggingface-hub",
    "insightface",
    "onnxruntime",
    "opencv-python",
    "pillow-heif",
    "Pillow",
    "PySide6",
    "python-dateutil",
    "rapidfuzz",
    "safetensors",
    "torch",
    "transformers",
    "ultralytics",
):
    optional_copy_metadata(distribution_name)

hiddenimports = sorted(set(hiddenimports))


a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="smart-photos",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="smart-photos",
)
