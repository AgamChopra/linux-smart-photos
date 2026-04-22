from __future__ import annotations

import argparse
from pathlib import Path

from .branding import APP_NAME
from .config import config_file_path, load_config
from .services.model_manager import ModelManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-photos-bootstrap",
        description=f"Bootstrap {APP_NAME} models and local configuration.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override the config file path.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Create config and directories without downloading AI models.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(f"{APP_NAME} config: {args.config or config_file_path()}")
    print(f"Media root: {config.media_root_path}")
    print(f"Model cache: {config.models_path}")

    if args.skip_models:
        print("Skipped AI model downloads.")
        return 0

    manager = ModelManager(config)
    paths = manager.download_recommended_models()
    if not paths:
        print("AI models already present.")
        return 0

    print("Installed AI models:")
    for path in paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
