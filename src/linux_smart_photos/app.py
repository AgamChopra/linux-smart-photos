from __future__ import annotations

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .branding import APP_DESKTOP_ID, APP_NAME, icon_path
from .config import load_config
from .services.library import LibraryService
from .ui.main_window import MainWindow
from .ui.theme import apply_app_theme


def main() -> int:
    config = load_config()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    QApplication.setDesktopFileName(APP_DESKTOP_ID)
    resolved_icon = icon_path()
    if resolved_icon.exists():
        app.setWindowIcon(QIcon(str(resolved_icon)))
    app.setStyle("Fusion")
    apply_app_theme(app, config)

    service = LibraryService(config)
    window = MainWindow(service)
    window.setWindowTitle(APP_NAME)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
