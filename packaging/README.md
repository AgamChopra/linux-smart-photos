# Packaging

This project ships two installation paths:

- repo-local source install via [smart-photos](/home/agam/Documents/github/linux-smart-photos/smart-photos)
- distributable AppImage built from a frozen PyInstaller bundle

## Build An AppImage

From the project root:

```bash
./packaging/build-appimage.sh
```

That script will:

1. create `.venv-packaging`
2. install the project plus `.[ai,packaging]`
3. freeze the launcher with PyInstaller
4. assemble an AppDir
5. fetch `appimagetool` from the official AppImage repository
6. emit `dist/Smart-Photos-<version>-<arch>.AppImage`

Options:

```bash
./packaging/build-appimage.sh --without-ai
./packaging/build-appimage.sh --skip-tool-download
```

## Notes

- The AppImage bundles the Python runtime and Python dependencies.
- AI model files are still stored outside the AppImage in the normal user data directory and downloaded on demand.
- For best cross-distro compatibility, build on an older supported Linux base in CI and run on newer systems.
