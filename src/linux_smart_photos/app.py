from __future__ import annotations

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .branding import APP_DESKTOP_ID, APP_NAME, icon_path
from .config import load_config
from .services.library import LibraryService
from .ui.main_window import MainWindow


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
    app.setStyleSheet(
        """
        QWidget {
            font-size: 13px;
        }
        QMainWindow {
            background: #faf7f2;
        }
        QTabWidget::pane {
            border: 0;
        }
        QListWidget, QTextEdit, QTableWidget {
            background: #fffdf9;
            border: 1px solid #ddd3c7;
            border-radius: 10px;
        }
        QPushButton, QComboBox, QLineEdit {
            min-height: 34px;
            border: 1px solid #d5c8b8;
            border-radius: 10px;
            padding: 4px 10px;
            background: #fffaf2;
        }
        QPushButton:hover {
            background: #f4ebe0;
        }
        """
    )

    service = LibraryService(config)
    window = MainWindow(service)
    window.setWindowTitle(APP_NAME)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
