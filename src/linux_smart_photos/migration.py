from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from .branding import APP_NAME
from .config import config_file_path, normalize_config_file, write_config
from .store import SQLiteLibraryStore


@dataclass(slots=True)
class MigrationResult:
    config_path: Path
    database_path: Path
    migrated_from: Path | None = None
    deleted_paths: list[Path] = field(default_factory=list)
    config_updated: bool = False


def migrate_configured_library(
    config_path: Path | None = None,
    *,
    delete_legacy: bool = True,
) -> MigrationResult:
    resolved_config_path = config_path or config_file_path()
    config, normalized = normalize_config_file(resolved_config_path)
    store = SQLiteLibraryStore(config.database_file)

    config_updated = normalized
    if config.database_file != store.path:
        config.database_path = str(store.path)
        write_config(config, resolved_config_path)
        config_updated = True

    deleted_paths: list[Path] = []
    if delete_legacy:
        deleted_paths = store.delete_legacy_json_files()

    return MigrationResult(
        config_path=resolved_config_path,
        database_path=store.path,
        migrated_from=store.migrated_from_legacy_json,
        deleted_paths=deleted_paths,
        config_updated=config_updated,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-photos-migrate",
        description=f"Migrate legacy {APP_NAME} JSON libraries to SQLite.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override the config file path.",
    )
    parser.add_argument(
        "--keep-legacy",
        action="store_true",
        help="Keep legacy JSON files after a successful migration.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = migrate_configured_library(args.config, delete_legacy=not bool(args.keep_legacy))

    print(f"{APP_NAME} config: {result.config_path}")
    print(f"SQLite library: {result.database_path}")
    if result.migrated_from is not None:
        print(f"Migrated legacy library: {result.migrated_from}")
    else:
        print("Migrated legacy library: none")
    print(f"Config updated: {'yes' if result.config_updated else 'no'}")
    if result.deleted_paths:
        print("Deleted legacy files:")
        for path in result.deleted_paths:
            print(f"- {path}")
    else:
        print("Deleted legacy files: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
