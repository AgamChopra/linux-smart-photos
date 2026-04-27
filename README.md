<p align="center">
  <img src="assets/smart-photos.svg" alt="Linux Smart Photos icon" width="256" height="256">
</p>

<h1 align="center">Linux Smart Photos</h1>

<p align="center">
  A private, local-first photo library for Linux with people, pets, search, albums, and memories.
</p>

<p align="center">
  <strong>Local media.</strong> <strong>Local database.</strong> <strong>Local AI.</strong>
</p>

---

Linux Smart Photos (LSP) helps you browse and organize personal photo libraries without sending your images to a cloud service. It indexes your media into a local SQLite database, generates thumbnails, detects people and pets, and gives you a clean everyday interface for reviewing matches and curating albums.

The default interface is a lightweight **Tauri** desktop app. The older **Qt Widgets** interface remains available as a diagnostics and fallback UI.

## Highlights

- Browse photos, videos, GIFs, and paired live photos.
- Search by text, tags, people, pets, year, media type, and favorites.
- Group detected human faces and pet faces into reviewable clusters.
- Confirm possible people/pet matches with a simple yes/no review flow.
- Create personas, albums, favorites, and memory collections.
- Keep thumbnails, model files, and library metadata on your machine.
- Use the Qt diagnostics UI when you want scan status, cluster internals, and lower-level controls.

## Interface Modes

| Mode | Command | Best For |
| --- | --- | --- |
| Tauri UI | `./smart-photos` or `./smart-photos --tauri` | Everyday use |
| Qt diagnostics | `./smart-photos --qt` | Debugging scans, models, and cluster status |
| CLI | `./smart-photos --cli status` | Scripts and quick checks |
| Electron legacy | `./smart-photos --electron` | Deprecated fallback only |

## Quick Start

Clone the project and launch it:

```bash
git clone <repository-url>
cd linux-smart-photos
./smart-photos
```

The launcher will create a virtual environment, install Python dependencies, initialize configuration, migrate older JSON data if needed, and install desktop integration.

To run setup without opening the app:

```bash
./setup-smart-photos
```

To force a full setup refresh:

```bash
./smart-photos --setup
```

## Requirements

Required:

- Linux
- Python 3.11 or newer

Recommended for the default Tauri interface:

- Node.js with `npm`
- Rust/Cargo
- WebKitGTK 4.1 and native build packages required by Tauri

If the Tauri UI cannot be built, LSP automatically falls back to the Qt diagnostics UI.

### Tauri Prerequisites

LSP vendors the Tauri project in `tauri/`, so you do not need to install a global Tauri CLI. You do need the system build dependencies, Node/npm, and Rust/Cargo before the default UI can be built.

These package names follow the current [Tauri v2 Linux prerequisites](https://v2.tauri.app/start/prerequisites/).

Debian / Ubuntu:

```bash
sudo apt update
sudo apt install -y nodejs npm \
  libwebkit2gtk-4.1-dev \
  build-essential \
  curl \
  wget \
  file \
  libxdo-dev \
  libssl-dev \
  libayatana-appindicator3-dev \
  librsvg2-dev

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
```

Fedora:

```bash
sudo dnf install -y nodejs npm \
  webkit2gtk4.1-devel \
  openssl-devel \
  curl \
  wget \
  file \
  libappindicator-gtk3-devel \
  librsvg2-devel \
  libxdo-devel
sudo dnf group install -y "c-development"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
```

Arch:

```bash
sudo pacman -Syu
sudo pacman -S --needed nodejs npm rustup \
  webkit2gtk-4.1 \
  base-devel \
  curl \
  wget \
  file \
  openssl \
  appmenu-gtk-module \
  libappindicator-gtk3 \
  librsvg \
  xdotool
rustup default stable
```

Confirm the tools are visible in your terminal:

```bash
node --version
npm --version
cargo --version
```

Then rebuild the app setup:

```bash
./setup-smart-photos --force
./smart-photos
```

If your distribution packages an old Node.js release and the Tauri build fails, install the current Node.js LTS release using your preferred Node version manager or distro-supported NodeSource package.

Setup builds the runnable Tauri binary only. Distribution packages are optional release artifacts:

```bash
cd tauri
npm run bundle -- --bundles deb rpm
```

AppImage bundling may require extra distro-specific `linuxdeploy`/FUSE support. If AppImage packaging fails after the binary is built, the app can still run normally.

If the Tauri window crashes with a Wayland protocol error, the launcher already disables the WebKitGTK dmabuf renderer by default. If your compositor still has issues, force X11 for one run:

```bash
SMART_PHOTOS_FORCE_X11=1 ./smart-photos
```

Optional for AI acceleration:

- NVIDIA GPU with working CUDA runtime
- TensorRT libraries if you want ONNX Runtime TensorRT acceleration

## AI Features

With the AI extras installed, LSP can:

- detect human faces, pets, pet faces, and common objects
- extract face and pet embeddings for grouping
- suggest likely persona matches
- improve matching after you confirm people and pets
- analyze sampled frames from videos and live photos

Recommended models can be installed from the **AI Models** view or via the CLI. Setup initializes the app and model cache but does not require you to download every model immediately.

## Everyday Workflow

1. Open Linux Smart Photos.
2. Let the library sync in the background.
3. Browse the Library view while thumbnails and AI results appear.
4. Open **Review Matches** to confirm or reject suggested face/pet clusters.
5. Create albums and mark favorites as you curate.
6. Use **Qt diagnostics** only when you need detailed scan or model information.

## Search Examples

Try queries like:

```text
cat
birthday
person:alice
pet:buddy
type:video
type:live_photo person:alice
year:2026 tag:dog
```

## CLI Examples

```bash
./smart-photos --cli status
./smart-photos --cli sync
./smart-photos --cli search cat --limit 10
./smart-photos --cli models status
./smart-photos --cli models install insightface_antelope
./smart-photos --cli migrate
```

## Data Locations

LSP stores app data in standard user directories.

| Data | Default Location |
| --- | --- |
| Config | `~/.config/linux-smart-photos/config.json` |
| SQLite library | `~/.local/share/linux-smart-photos/library.sqlite3` |
| Thumbnails | `~/.cache/linux-smart-photos/thumbnails` |
| Cluster previews | `~/.cache/linux-smart-photos/cluster-previews` |
| Model cache | `~/.local/share/linux-smart-photos/models` |

Older `library.json` files are migrated to SQLite during setup. After a successful migration, the old JSON library is removed.

You can use [config.example.json](config.example.json) as a template for custom paths and performance settings.

## Performance Notes

- The gallery keeps a bounded decoded-thumbnail RAM cache.
- `gallery_thumbnail_cache_mb` controls the thumbnail RAM budget.
- `gallery_prefetch_all_thumbnails` warms the active gallery in idle slices.
- Video AI scans sampled frames rather than every frame.
- Face and pet clustering is stored incrementally in SQLite so work can resume after restarts.

## Recommended Models

LSP can use these models:

- Object detection: [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/)
- Human face detector: [InsightFace SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/README.md)
- Human face embeddings: [InsightFace buffalo_sc](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)
- Pet personas: [AvitoTech DINOv2-Small Animal Identification](https://huggingface.co/AvitoTech/DINO-v2-small-for-animal-identification)
- Pet face detector: [LostPetInitiative yolov7-pet-face.pt](https://zenodo.org/records/7607110)

On systems where ONNX Runtime exposes `TensorrtExecutionProvider`, LSP prefers TensorRT for the human face detector and falls back to CUDA or CPU if TensorRT is unavailable.

## Current Limits

- Tauri requires system WebKitGTK support on Linux.
- The Qt interface is intentionally more technical and diagnostic-oriented.
- Video preview depends on available system media support.
- Video AI analysis is sampled.
- Memories are currently rule-based.
- Some upstream InsightFace model packs are published for non-commercial research use; check upstream model licenses before commercial use.

## License

Linux Smart Photos is released under the [MIT License](LICENSE).
