from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from ..config import AppConfig


LIGHT_PALETTE = {
    "background": "#faf7f2",
    "surface": "#fffdf9",
    "surface_alt": "#f3f1ec",
    "text": "#211f1c",
    "muted": "#6f665c",
    "border": "#ddd3c7",
    "control": "#fffaf2",
    "control_hover": "#f4ebe0",
    "accent": "#0d3b66",
}

DARK_PALETTE = {
    "background": "#151716",
    "surface": "#20231f",
    "surface_alt": "#292d28",
    "text": "#f2efe8",
    "muted": "#beb6aa",
    "border": "#3c423a",
    "control": "#272b26",
    "control_hover": "#333932",
    "accent": "#7fc7a5",
}


def default_palette(theme: str) -> dict[str, str]:
    return DARK_PALETTE if theme == "dark" else LIGHT_PALETTE


def normalize_theme(theme: str) -> str:
    return theme if theme in {"light", "dark"} else "light"


def valid_color(value: str, fallback: str) -> str:
    color = QColor(value)
    if color.isValid():
        return color.name()
    return QColor(fallback).name()


def readable_text_color(background: str) -> str:
    color = QColor(background)
    if not color.isValid():
        return "#ffffff"
    luminance = (
        0.2126 * color.redF()
        + 0.7152 * color.greenF()
        + 0.0722 * color.blueF()
    )
    return "#111111" if luminance > 0.58 else "#ffffff"


def build_app_stylesheet(config: AppConfig) -> str:
    theme = normalize_theme(getattr(config, "ui_theme", "light"))
    palette = dict(default_palette(theme))
    palette["accent"] = valid_color(
        getattr(config, "ui_accent_color", "") or palette["accent"],
        palette["accent"],
    )
    custom_background = getattr(config, "ui_background_color", "") or ""
    if custom_background:
        palette["background"] = valid_color(custom_background, palette["background"])
        background_color = QColor(palette["background"])
        if theme == "dark":
            palette["surface"] = background_color.lighter(122).name()
            palette["surface_alt"] = background_color.lighter(145).name()
            palette["control"] = background_color.lighter(135).name()
            palette["control_hover"] = background_color.lighter(158).name()
            palette["border"] = background_color.lighter(190).name()
        else:
            palette["surface"] = background_color.lighter(106).name()
            palette["surface_alt"] = background_color.darker(104).name()
            palette["control"] = background_color.lighter(110).name()
            palette["control_hover"] = background_color.darker(106).name()
            palette["border"] = background_color.darker(122).name()
        palette["text"] = readable_text_color(palette["background"])
        palette["muted"] = QColor(palette["text"]).darker(130).name() if theme == "light" else QColor(palette["text"]).darker(115).name()

    accent_text = readable_text_color(palette["accent"])

    return f"""
        QWidget {{
            background: {palette["background"]};
            color: {palette["text"]};
            font-size: 13px;
            selection-background-color: {palette["accent"]};
            selection-color: {accent_text};
        }}
        QMainWindow, QDialog {{
            background: {palette["background"]};
        }}
        QStatusBar, QMenuBar {{
            background: {palette["surface"]};
            color: {palette["text"]};
        }}
        QMenuBar::item:selected, QMenu::item:selected {{
            background: {palette["control_hover"]};
        }}
        QMenu {{
            background: {palette["surface"]};
            color: {palette["text"]};
            border: 1px solid {palette["border"]};
        }}
        QTabWidget::pane {{
            border: 0;
        }}
        QTabBar::tab {{
            background: {palette["control"]};
            border: 1px solid {palette["border"]};
            border-bottom: 0;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 8px 12px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{
            background: {palette["surface"]};
            color: {palette["text"]};
        }}
        QListWidget, QTextEdit, QTableWidget, QTreeWidget, QLineEdit, QComboBox, QSpinBox {{
            background: {palette["surface"]};
            color: {palette["text"]};
            border: 1px solid {palette["border"]};
            border-radius: 10px;
        }}
        QListWidget::item:selected, QTableWidget::item:selected {{
            background: {palette["accent"]};
            color: {accent_text};
        }}
        QLabel#PreviewPanel, QLabel#VideoPreviewPanel, QLabel#FaceMapPreview {{
            background: {palette["surface_alt"]};
            color: {palette["text"]};
            border: 1px solid {palette["border"]};
            border-radius: 12px;
        }}
        QPushButton, QComboBox, QLineEdit, QSpinBox {{
            min-height: 34px;
            border: 1px solid {palette["border"]};
            border-radius: 10px;
            padding: 4px 10px;
            background: {palette["control"]};
            color: {palette["text"]};
        }}
        QPushButton:hover, QComboBox:hover, QLineEdit:hover, QSpinBox:hover {{
            background: {palette["control_hover"]};
        }}
        QPushButton:pressed {{
            background: {palette["accent"]};
            color: {accent_text};
        }}
        QPushButton:disabled, QComboBox:disabled, QLineEdit:disabled {{
            color: {palette["muted"]};
        }}
        QProgressBar {{
            background: {palette["surface_alt"]};
            border: 1px solid {palette["border"]};
            border-radius: 8px;
            text-align: center;
            color: {palette["text"]};
        }}
        QProgressBar::chunk {{
            background: {palette["accent"]};
            border-radius: 7px;
        }}
        QSlider::groove:horizontal {{
            height: 6px;
            background: {palette["surface_alt"]};
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            width: 18px;
            margin: -6px 0;
            border-radius: 9px;
            background: {palette["accent"]};
        }}
        QScrollBar:vertical, QScrollBar:horizontal {{
            background: {palette["background"]};
            border: 0;
        }}
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
            background: {palette["border"]};
            border-radius: 6px;
            min-height: 32px;
            min-width: 32px;
        }}
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
            background: {palette["accent"]};
        }}
    """


def apply_app_theme(app: QApplication, config: AppConfig) -> None:
    app.setStyleSheet(build_app_stylesheet(config))
