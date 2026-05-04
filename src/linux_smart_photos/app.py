from __future__ import annotations

import argparse
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .branding import APP_DESKTOP_ID, APP_NAME, icon_path
from .config import load_config
from .services.library import DEFAULT_SYNC_ITEM_LIMIT, LibraryService
from .ui.main_window import MainWindow
from .ui.theme import apply_app_theme


def _parse_gui_args(argv: list[str]) -> tuple[int | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sync-limit", default=str(DEFAULT_SYNC_ITEM_LIMIT))
    parser.add_argument("--sync-full", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    if bool(args.sync_full):
        return None, remaining
    value = str(args.sync_limit).strip().lower()
    if value in {"full", "all"}:
        return None, remaining
    try:
        return max(1, int(value)), remaining
    except ValueError as exc:
        raise SystemExit('--sync-limit must be a positive integer or "full".') from exc


def main() -> int:
    sync_item_limit, qt_args = _parse_gui_args(sys.argv[1:])
    config = load_config()

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    QApplication.setDesktopFileName(APP_DESKTOP_ID)
    resolved_icon = icon_path()
    if resolved_icon.exists():
        app.setWindowIcon(QIcon(str(resolved_icon)))
    app.setStyle("Fusion")
    apply_app_theme(app, config)

    service = LibraryService(config)
    window = MainWindow(service, sync_item_limit=sync_item_limit)
    window.setWindowTitle(APP_NAME)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
