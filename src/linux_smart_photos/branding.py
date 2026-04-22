from __future__ import annotations

import sys
from pathlib import Path


APP_NAME = "Smart Photos"
APP_DESCRIPTION = "Browse, search, and organize your local photo library."
APP_DESKTOP_ID = "smart-photos"
APP_ICON_NAME = "smart-photos"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resource_root() -> Path:
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        return Path(meipass).resolve()
    return project_root()


def icon_path() -> Path:
    return resource_root() / "assets" / "smart-photos.svg"
