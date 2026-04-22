# Smart Photos

Smart Photos is a Linux photo library app for browsing, searching, and organizing personal media collections.

It supports:

- Photos, videos, GIFs, and paired live photos
- Albums, favorites, people, and pets
- Search by text, tags, people, pets, media type, and favorites
- Face and pet recognition with manual correction
- Auto-generated memories
- Inline preview for videos, GIFs, and live-photo motion clips

The app is designed for photo libraries stored in a structure like:

```text
file/photos/<year>/<date>/<asset>
```

Changes in the library are detected automatically when the app starts. New items are indexed, deleted items are removed, and saved people, pet, album, and memory data are kept in a local JSON database for fast loading.

## AI Features

With the AI extras installed, Smart Photos can:

- detect people, pets, and common objects
- suggest matches for existing people and pet profiles
- improve future matches as you confirm more photos
- prioritize cat detection in the pet pipeline
- analyze sampled frames from videos and live photos

Recommended model downloads are available in the `AI Models` tab.

## Install

The easiest way to start is with the wrapper launcher. On first run it will:

- create `.venv`
- install the app and AI dependencies
- download the recommended AI models
- install a desktop launcher and icon in `~/.local/share/applications`
- symlink the launcher to `~/.local/bin/smart-photos`

Clone the repository and run:

```bash
git clone <repository-url>
cd linux-smart-photos
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

## Files

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

You can use [config.example.json](config.example.json) as a template.

## Desktop Integration

The setup script installs:

- a desktop launcher named `Smart Photos`
- the app icon under `~/.local/share/icons/hicolor/scalable/apps/smart-photos.svg`
- a desktop entry at `~/.local/share/applications/smart-photos.desktop`

If an AppImage release is available, it can be used as a portable Linux build.

## Recommended Models

Smart Photos includes download links for these models:

- Object detection: [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/)
- Human face personas: [InsightFace `buffalo_sc`](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)
- Pet personas: [AvitoTech DINOv2-Small Animal Identification](https://huggingface.co/AvitoTech/DINO-v2-small-for-animal-identification)
- Pet face detector: [LostPetInitiative `yolov7-pet-face.pt`](https://zenodo.org/records/7607110)

## Search Examples

- `cat`
- `birthday`
- `person:alice`
- `pet:buddy`
- `type:video`
- `type:live_photo person:alice`
- `year:2026 tag:dog`

## Current Limits

- Video preview depends on the Qt multimedia backend present on the machine.
- Video AI analysis is sampled instead of scanning every frame.
- The memory system is rule-based.
- InsightFace model packs are listed upstream as non-commercial research models.
