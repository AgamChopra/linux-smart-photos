# Linux Smart Photos

Python desktop photo library for Linux with:

- Incremental filesystem sync against `file/photos/<year>/<date>/<asset>`
- JSON-backed metadata for fast startup lookups
- Support for images, videos, GIFs, and paired live photos
- Albums, favorites, people/pets views, search, and auto-generated memories
- Manual correction flow for unrecognized people and pets
- Inline video preview, AI model management, cat-priority pet detection, learned pet personas, and sampled video AI analysis

## What This App Does

At launch the app walks the configured media root, detects new or removed files, and updates a single JSON library database. New or changed assets are analyzed for metadata, thumbnails, faces, and optional object tags. Manual persona corrections are also persisted in that JSON database so the next launch can reuse them immediately.

The current build is designed as a strong local-first foundation rather than a clone of every iPhone Photos feature byte-for-byte. The UI and service layer are in place for:

- Library browsing with thumbnails and metadata preview
- Search over titles, paths, tags, detected labels, notes, and persona names
- Filters by media type, people, pets, and favorites
- Manual album creation and adding selected items to albums
- Persona creation and correction for detected regions or whole assets
- Inline preview for videos, GIFs, and live-photo motion clips
- Memory generation by day, month, multi-day spans, personas, persona pairs, themes, and favorites
- AI model status and downloads from an in-app `AI Models` tab
- Interval-sampled AI analysis for videos and live-photo motion clips

## AI Features

The app runs without the optional AI stack, but the iPhone-style recognition features improve sharply when you install it:

- `insightface` + `onnxruntime`: human face detection and persona embeddings
- `ultralytics`: object detection with `YOLO11n`
- `transformers` + `torch` + `huggingface-hub`: learned pet persona embeddings
- `pillow-heif`: HEIC/HEIF decoding

Persona matching already uses a multi-reference strategy:

- Each manual face or pet correction adds another embedding exemplar to that persona.
- New detections are compared against the full reference set, not only one anchor photo.
- Automatic matching stays conservative because only manual confirmations expand the stored reference set.

Without those optional dependencies, the app still syncs the library, builds thumbnails for supported formats, manages albums, favorites, manual personas, and memories.

## Install

The easiest end-user path is the wrapper launcher. On first run it will:

- create `.venv`
- install the app and AI dependencies
- download the recommended AI models
- install a desktop launcher and icon in `~/.local/share/applications`
- symlink the launcher to `~/.local/bin/smart-photos`

Run:

```bash
./smart-photos
```

If you want to bootstrap without launching the GUI:

```bash
./setup-smart-photos
```

The first run can take a while because it installs Python packages and model files.

## Manual Install

Create a virtual environment and install the base app:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the AI extras if you want human face personas, object detection, and learned pet personas:

```bash
pip install -e .[ai]
```

Notes:

- The app can download recommended models from the `AI Models` tab.
- `torch`, `insightface`, and `onnxruntime` make the AI install noticeably larger.
- Inline video playback depends on Qt multimedia support being available in your environment.

## Run

```bash
./smart-photos
```

Or:

```bash
./smart-photos --gui
```

CLI mode:

```bash
./smart-photos --cli status
./smart-photos --cli sync
./smart-photos --cli search cat --limit 10
./smart-photos --cli models status
```

On first launch the app writes config here:

```text
~/.config/linux-smart-photos/config.json
```

The JSON library database lives here by default:

```text
~/.local/share/linux-smart-photos/library.json
```

Thumbnail cache lives here by default:

```text
~/.cache/linux-smart-photos/thumbnails
```

Model cache lives here by default:

```text
~/.local/share/linux-smart-photos/models
```

You can use [config.example.json](/home/agam/Documents/github/linux-smart-photos/config.example.json) as a template.

## Desktop Integration

The setup script installs:

- a desktop launcher named `Smart Photos`
- the app icon under `~/.local/share/icons/hicolor/scalable/apps/smart-photos.svg`
- a desktop entry at `~/.local/share/applications/smart-photos.desktop`

The desktop launcher points at the repo-local wrapper script, so updates to the repo keep using the same launch target.

## AppImage Shipping

For distro-independent distribution, this repo now includes a frozen AppImage build path.

Build locally:

```bash
./packaging/build-appimage.sh
```

That produces:

```text
dist/Smart-Photos-<version>-<arch>.AppImage
```

Useful variants:

```bash
./packaging/build-appimage.sh --without-ai
./packaging/build-appimage.sh --skip-tool-download
```

Notes:

- the AppImage bundles Python and the installed Python dependencies
- AI model files are still stored in the normal user data directory and downloaded outside the AppImage
- GitHub Actions now includes an AppImage build workflow at [.github/workflows/build-appimage.yml](/home/agam/Documents/github/linux-smart-photos/.github/workflows/build-appimage.yml)

## Recommended Models

The app ships with built-in metadata and download links for these defaults:

- Object detection: [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/)
- Human face personas: [InsightFace `buffalo_sc`](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)
- Pet personas: [AvitoTech DINOv2-Small Animal Identification](https://huggingface.co/AvitoTech/DINO-v2-small-for-animal-identification)
- Pet face detector: [LostPetInitiative `yolov7-pet-face.pt`](https://zenodo.org/records/7607110)

Cat priority:

- Cats are treated as the higher-priority pet class in the current pipeline.
- The dedicated pet-face model is now used directly when it is installed.
- Cat detections use a lower detector threshold than dogs.
- A cat-face fallback detector is still used when the dedicated detector misses a cat.
- Pet personas now use learned embeddings instead of simple perceptual hashes.

## Search Examples

- `cat`
- `birthday`
- `person:alice`
- `pet:buddy`
- `type:video`
- `type:live_photo person:alice`
- `year:2026 tag:dog`

## Project Layout

- [main.py](/home/agam/Documents/github/linux-smart-photos/main.py)
- [src/linux_smart_photos/app.py](/home/agam/Documents/github/linux-smart-photos/src/linux_smart_photos/app.py)
- [src/linux_smart_photos/services/library.py](/home/agam/Documents/github/linux-smart-photos/src/linux_smart_photos/services/library.py)
- [src/linux_smart_photos/services/vision.py](/home/agam/Documents/github/linux-smart-photos/src/linux_smart_photos/services/vision.py)
- [src/linux_smart_photos/ui/main_window.py](/home/agam/Documents/github/linux-smart-photos/src/linux_smart_photos/ui/main_window.py)

## Current Limits

- Video preview depends on the Qt multimedia backend present on the machine.
- Video AI analysis is sampled and deduplicated for efficiency rather than frame-by-frame exhaustive.
- The memory system is still heuristic, though it is much richer than the initial month/persona/favorites-only version.
- InsightFace model packs are listed upstream as non-commercial research models.

The important part is that the architecture is already set up so those can be expanded without replacing the library model or UI shell.
