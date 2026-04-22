#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGING_VENV="$ROOT_DIR/.venv-packaging"
TOOLS_DIR="$ROOT_DIR/packaging/tools"
PYINSTALLER_SPEC="$ROOT_DIR/packaging/pyinstaller/smart-photos.spec"
PYINSTALLER_DIST="$ROOT_DIR/dist"
PYINSTALLER_BUILD="$ROOT_DIR/build/pyinstaller"
APPIMAGE_BUILD="$ROOT_DIR/build/appimage"
APPDIR="$APPIMAGE_BUILD/Smart-Photos.AppDir"
ARCH="${ARCH:-$(uname -m)}"
WITH_AI="1"
DOWNLOAD_TOOLS="1"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: ./packaging/build-appimage.sh [--with-ai] [--without-ai] [--skip-tool-download] [--help]

Builds a PyInstaller onedir bundle and wraps it into an AppImage.
EOF
}

while (($#)); do
  case "$1" in
    --with-ai)
      WITH_AI="1"
      shift
      ;;
    --without-ai)
      WITH_AI="0"
      shift
      ;;
    --skip-tool-download)
      DOWNLOAD_TOOLS="0"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

create_packaging_env() {
  "$PYTHON_BIN" -m venv "$PACKAGING_VENV"
  "$PACKAGING_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
  if [[ "$WITH_AI" == "1" ]]; then
    "$PACKAGING_VENV/bin/pip" install -e "$ROOT_DIR[ai,packaging]"
  else
    "$PACKAGING_VENV/bin/pip" install -e "$ROOT_DIR[packaging]"
  fi
}

download_appimagetool() {
  local target="$TOOLS_DIR/appimagetool-$ARCH.AppImage"
  local url="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage"

  if command -v appimagetool >/dev/null 2>&1; then
    command -v appimagetool
    return 0
  fi

  if [[ -x "$target" ]]; then
    printf '%s\n' "$target"
    return 0
  fi

  if [[ "$DOWNLOAD_TOOLS" != "1" ]]; then
    echo "Missing $target and tool download disabled." >&2
    exit 1
  fi

  mkdir -p "$TOOLS_DIR"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$target"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$target" "$url"
  else
    echo "curl or wget is required to download appimagetool." >&2
    exit 1
  fi
  chmod +x "$target"
  printf '%s\n' "$target"
}

project_version="$("$PYTHON_BIN" - "$ROOT_DIR" <<'PY'
from pathlib import Path
import tomllib
import sys

root = Path(sys.argv[1])
payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
print(payload["project"]["version"])
PY
)"

mkdir -p "$TOOLS_DIR" "$APPIMAGE_BUILD" "$ROOT_DIR/dist"
create_packaging_env

echo "Building PyInstaller bundle"
"$PACKAGING_VENV/bin/pyinstaller" \
  --noconfirm \
  --clean \
  --distpath "$PYINSTALLER_DIST" \
  --workpath "$PYINSTALLER_BUILD" \
  "$PYINSTALLER_SPEC"

FROZEN_DIR="$PYINSTALLER_DIST/smart-photos"
FROZEN_BIN="$FROZEN_DIR/smart-photos"
if [[ ! -x "$FROZEN_BIN" ]]; then
  echo "Expected frozen binary not found: $FROZEN_BIN" >&2
  exit 1
fi

echo "Preparing AppDir"
rm -rf "$APPDIR"
mkdir -p \
  "$APPDIR/usr/lib" \
  "$APPDIR/usr/share/applications" \
  "$APPDIR/usr/share/icons/hicolor/scalable/apps"
cp -R "$FROZEN_DIR" "$APPDIR/usr/lib/smart-photos"
install -m 0755 "$ROOT_DIR/packaging/appimage/AppRun" "$APPDIR/AppRun"
install -m 0644 "$ROOT_DIR/packaging/appimage/smart-photos.desktop" "$APPDIR/smart-photos.desktop"
install -m 0644 "$ROOT_DIR/packaging/appimage/smart-photos.desktop" "$APPDIR/usr/share/applications/smart-photos.desktop"
install -m 0644 "$ROOT_DIR/assets/smart-photos.svg" "$APPDIR/smart-photos.svg"
install -m 0644 "$ROOT_DIR/assets/smart-photos.svg" "$APPDIR/usr/share/icons/hicolor/scalable/apps/smart-photos.svg"

APPIMAGETOOL_BIN="$(download_appimagetool)"
OUTPUT_FILE="$ROOT_DIR/dist/Smart-Photos-$project_version-$ARCH.AppImage"

echo "Creating AppImage"
APPIMAGE_EXTRACT_AND_RUN=1 \
APPIMAGETOOL_APP_NAME="Smart Photos" \
VERSION="$project_version" \
ARCH="$ARCH" \
  "$APPIMAGETOOL_BIN" "$APPDIR" "$OUTPUT_FILE"

echo "Built $OUTPUT_FILE"
