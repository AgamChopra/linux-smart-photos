from __future__ import annotations

import argparse
from pathlib import Path

from .branding import APP_NAME
from .config import config_file_path, load_config
from .migration import migrate_configured_library


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-photos-cli",
        description=f"Command-line tools for {APP_NAME}.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override the config file path.",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Show library and model status.")

    sync_parser = subparsers.add_parser("sync", help="Scan the photo library for changes.")
    sync_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the sync summary as JSON-like lines.",
    )

    search_parser = subparsers.add_parser("search", help="Search the indexed library.")
    search_parser.add_argument("query", nargs="*", help="Search query text.")
    search_parser.add_argument("--type", default="all", help="Media kind filter.")
    search_parser.add_argument("--persona-kind", default="all", help="Filter by person or pet.")
    search_parser.add_argument("--persona-id", default="", help="Restrict to one persona id.")
    search_parser.add_argument("--favorites", action="store_true", help="Only show favorites.")
    search_parser.add_argument("--limit", type=int, default=20, help="Maximum items to print.")

    models_parser = subparsers.add_parser("models", help="Inspect or download AI models.")
    model_subparsers = models_parser.add_subparsers(dest="models_command")
    model_subparsers.add_parser("status", help="Show installed model status.")
    install_parser = model_subparsers.add_parser("install", help="Download AI models.")
    install_parser.add_argument(
        "model_ids",
        nargs="*",
        help="Optional model ids. If omitted, installs the recommended set.",
    )

    migrate_parser = subparsers.add_parser("migrate", help="Migrate legacy JSON library data to SQLite.")
    migrate_parser.add_argument(
        "--keep-legacy",
        action="store_true",
        help="Keep legacy JSON files after a successful migration.",
    )

    subparsers.add_parser("gui", help="Launch the desktop UI.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "status"

    if command == "gui":
        from .app import main as gui_main

        return gui_main()
    if command == "migrate":
        return run_migrate(args.config, keep_legacy=bool(args.keep_legacy))

    config = load_config(args.config)
    from .services.library import LibraryService

    service = LibraryService(config)

    if command == "status":
        return run_status(service, args.config)
    if command == "sync":
        return run_sync(service, json_output=bool(args.json))
    if command == "search":
        return run_search(service, args)
    if command == "models":
        return run_models(service, getattr(args, "models_command", None), getattr(args, "model_ids", []))

    parser.print_help()
    return 1


def run_status(service: LibraryService, config_override: Path | None) -> int:
    print(APP_NAME)
    print(f"Config: {config_override or config_file_path()}")
    print(f"Media root: {service.config.media_root_path}")
    print(f"Items: {len(service.list_items())}")
    print(f"Personas: {len(service.list_personas())}")
    print(f"Albums: {len(service.list_albums())}")
    print(f"Memories: {len(service.list_memories())}")
    print("Models:")
    for status in service.model_statuses():
        state = "installed" if status.installed else "missing"
        print(f"- {status.id}: {state}")
    return 0


def run_sync(service: LibraryService, json_output: bool = False) -> int:
    summary = service.sync()
    if json_output:
        print("{")
        print(f'  "added": {summary.added},')
        print(f'  "updated": {summary.updated},')
        print(f'  "removed": {summary.removed}')
        print("}")
    else:
        print(f"Sync complete: added={summary.added} updated={summary.updated} removed={summary.removed}")
    return 0


def run_search(service: LibraryService, args: argparse.Namespace) -> int:
    query = " ".join(args.query).strip()
    items = service.search_items(
        query=query,
        media_kind=str(args.type),
        persona_kind=str(args.persona_kind),
        persona_id=str(args.persona_id),
        favorites_only=bool(args.favorites),
    )

    if not items:
        print("No items matched.")
        return 0

    for item in items[: max(1, int(args.limit))]:
        print(f"{item.id}  {item.media_kind:10}  {item.captured_at}  {item.path}")
    return 0


def run_models(service: LibraryService, models_command: str | None, model_ids: list[str]) -> int:
    if not models_command or models_command == "status":
        for status in service.model_statuses():
            state = "installed" if status.installed else "missing"
            print(f"{status.id:28}  {state:9}  {status.local_path}")
        return 0

    installed_paths: list[str] = []
    if model_ids:
        for model_id in model_ids:
            installed_paths.append(service.download_model(model_id))
    else:
        installed_paths = service.download_recommended_models()

    print("Installed models:")
    for path in installed_paths:
        print(f"- {path}")
    return 0


def run_migrate(config_override: Path | None, *, keep_legacy: bool = False) -> int:
    result = migrate_configured_library(config_override, delete_legacy=not keep_legacy)
    print(f"{APP_NAME} migration")
    print(f"Config: {config_override or config_file_path()}")
    print(f"Database: {result.database_path}")
    if result.migrated_from is not None:
        print(f"Migrated from: {result.migrated_from}")
    else:
        print("Migrated from: none")
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
