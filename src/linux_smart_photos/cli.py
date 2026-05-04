from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from time import monotonic

from .branding import APP_NAME
from .config import config_file_path, load_config
from .migration import migrate_configured_library

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - exercised when tqdm is unavailable.
    tqdm = None


DEFAULT_CLI_SYNC_ITEM_LIMIT = 1000


def ensure_tqdm_available() -> bool:
    global tqdm
    if tqdm is not None:
        return True
    print("tqdm is not installed; installing it for CLI progress bars...", file=sys.stderr)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "tqdm>=4.66"],
            stdout=subprocess.DEVNULL,
        )
        module = importlib.import_module("tqdm")
        tqdm = module.tqdm
        return True
    except Exception as exc:
        print(f"Unable to auto-install tqdm: {exc}", file=sys.stderr)
        print("Continuing with plain progress output.", file=sys.stderr)
        return False


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
    sync_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Run all sync stages with yes answers and no prompts.",
    )
    sync_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Print plain status lines instead of progress bars.",
    )
    sync_parser.add_argument(
        "--limit",
        default=str(DEFAULT_CLI_SYNC_ITEM_LIMIT),
        help='Latest item count to examine, or "full". Default: 1000.',
    )
    sync_parser.add_argument(
        "--full",
        action="store_true",
        help='Examine the full library. Equivalent to "--limit full".',
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
        return run_sync(service, args)
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
    print(f"Compute mode: {service.config.compute_mode}")
    print(f"Items: {len(service.list_items())}")
    print(f"Personas: {len(service.list_personas())}")
    print(f"Albums: {len(service.list_albums())}")
    print(f"Memories: {len(service.list_memories())}")
    print("Models:")
    for status in service.model_statuses():
        state = "installed" if status.installed else "missing"
        print(f"- {status.id}: {state}")
    return 0


def run_sync(service: LibraryService, args: argparse.Namespace) -> int:
    sync_item_limit = parse_sync_item_limit(args)
    if bool(args.json):
        summary = service.sync(sync_item_limit=sync_item_limit)
        print("{")
        print(f'  "added": {summary.added},')
        print(f'  "updated": {summary.updated},')
        print(f'  "removed": {summary.removed}')
        print("}")
        return 0

    if bool(args.yes):
        choices = SyncCliChoices(
            scan_library=True,
            apply_ai=True,
            detect_objects=True,
            detect_people=True,
            detect_pets=True,
            cluster_unassigned=True,
        )
    elif sys.stdin.isatty():
        choices = prompt_sync_choices()
    else:
        choices = SyncCliChoices(
            scan_library=True,
            apply_ai=True,
            detect_objects=True,
            detect_people=True,
            detect_pets=False,
            cluster_unassigned=True,
        )

    if not any((choices.scan_library, choices.apply_ai, choices.cluster_unassigned)):
        print("No sync stages selected.")
        return 0

    print(f"Scope: {format_sync_scope(sync_item_limit)}")
    progress_factory = PlainProgressRenderer
    if not bool(args.no_progress) and ensure_tqdm_available():
        progress_factory = TqdmProgressRenderer
    stage_results: list[tuple[str, object]] = []

    if choices.scan_library:
        print_stage_header("1. Scanning library and generating thumbnails")
        renderer = progress_factory("scan")
        summary = service.sync(
            progress_callback=renderer,
            include_pets=False,
            detect_objects=False,
            detect_people=False,
            scan_only=True,
            sync_item_limit=sync_item_limit,
        )
        renderer.close()
        stage_results.append(("scan", summary))
        print(f"Scan complete: added={summary.added} updated={summary.updated} removed={summary.removed}")

    if choices.apply_ai:
        print_stage_header("2. Applying AI detections")
        details = []
        if choices.detect_objects:
            details.append("objects")
        if choices.detect_people:
            details.append("people")
        if choices.detect_pets:
            details.append("pets")
        print(f"Detection targets: {', '.join(details) if details else 'none'}")
        if details:
            renderer = progress_factory("ai")
            summary = service.sync(
                progress_callback=renderer,
                include_pets=choices.detect_pets,
                detect_objects=choices.detect_objects,
                detect_people=choices.detect_people,
                scan_only=False,
                sync_item_limit=sync_item_limit,
            )
            renderer.close()
            stage_results.append(("ai", summary))
            print(f"AI detection complete: added={summary.added} updated={summary.updated} removed={summary.removed}")
        else:
            print("Skipped AI detection because no detection targets were selected.")

    if choices.cluster_unassigned:
        print_stage_header("3. Clustering unassigned detections")
        renderer = progress_factory("clusters")
        assigned = service.assign_unclustered_detections_to_known_personas(
            include_pets=choices.detect_pets,
            progress_callback=renderer,
            sync_item_limit=sync_item_limit,
        )
        service.rebuild_unknown_cluster_caches(
            partial=False,
            include_pets=choices.detect_pets,
            progress_callback=renderer,
            sync_item_limit=sync_item_limit,
        )
        renderer.close()
        person_clusters = service.list_unknown_persona_clusters(kind="person")
        pet_clusters = service.list_unknown_persona_clusters(kind="pet") if choices.detect_pets else []
        stage_results.append((
            "clusters",
            {"known_assigned": assigned, "person": len(person_clusters), "pet": len(pet_clusters)},
        ))
        print(
            "Cluster scan complete: "
            f"known_assigned={assigned} "
            f"person_clusters={len(person_clusters)} pet_clusters={len(pet_clusters)}"
        )

    print_stage_header("Summary")
    for stage_name, result in stage_results:
        if hasattr(result, "added"):
            print(
                f"{stage_name}: "
                f"added={result.added} updated={result.updated} removed={result.removed}"
            )
        else:
            print(f"{stage_name}: {result}")
    return 0


def parse_sync_item_limit(args: argparse.Namespace) -> int | None:
    if bool(getattr(args, "full", False)):
        return None
    value = str(getattr(args, "limit", DEFAULT_CLI_SYNC_ITEM_LIMIT)).strip().lower()
    if value in {"full", "all"}:
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise SystemExit('sync --limit must be a positive integer or "full".') from exc
    if limit < 1:
        raise SystemExit('sync --limit must be a positive integer or "full".')
    return limit


def format_sync_scope(sync_item_limit: int | None) -> str:
    if sync_item_limit is None:
        return "full library"
    return f"latest {sync_item_limit} media items"


class SyncCliChoices:
    def __init__(
        self,
        *,
        scan_library: bool,
        apply_ai: bool,
        detect_objects: bool,
        detect_people: bool,
        detect_pets: bool,
        cluster_unassigned: bool,
    ) -> None:
        self.scan_library = scan_library
        self.apply_ai = apply_ai
        self.detect_objects = detect_objects
        self.detect_people = detect_people
        self.detect_pets = detect_pets
        self.cluster_unassigned = cluster_unassigned


def prompt_sync_choices() -> SyncCliChoices:
    print(APP_NAME)
    print("Interactive CLI sync")
    scan_library = prompt_yes_no(
        "1) Scan library for new/changed photos and generate thumbnails?",
        default=True,
    )
    apply_ai = prompt_yes_no(
        "2) Apply object/person/pet detection after scanning?",
        default=True,
    )
    detect_objects = False
    detect_people = False
    detect_pets = False
    if apply_ai:
        detect_objects = prompt_yes_no("   Detect objects?", default=True)
        detect_people = prompt_yes_no("   Detect people/faces?", default=True)
        detect_pets = prompt_yes_no("   Detect pets? This is slower and optional.", default=False)
        if not any((detect_objects, detect_people, detect_pets)):
            apply_ai = False
    cluster_unassigned = prompt_yes_no(
        "3) Add unassigned detections to known/unknown clusters?",
        default=apply_ai or scan_library,
    )
    return SyncCliChoices(
        scan_library=scan_library,
        apply_ai=apply_ai,
        detect_objects=detect_objects,
        detect_people=detect_people,
        detect_pets=detect_pets,
        cluster_unassigned=cluster_unassigned,
    )


def prompt_yes_no(question: str, *, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {suffix} ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def print_stage_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


class PlainProgressRenderer:
    def __init__(self, label: str) -> None:
        self.label = label
        self.last_line = ""
        self.last_width = 0
        self.last_emit_at = 0.0
        self.stream = sys.stderr
        self.overwrite = self.stream.isatty()

    def __call__(self, update) -> None:
        detail = format_progress_detail(update)
        line = f"{update.message}"
        if update.total > 0:
            line += f" [{update.current}/{update.total}]"
        if detail:
            line += f" | {detail}"
        if line != self.last_line:
            now = monotonic()
            finished = update.total > 0 and update.current >= update.total
            if not self.overwrite and not finished and now - self.last_emit_at < 0.75:
                return
            self.last_emit_at = now
            if self.overwrite:
                width = max(20, shutil.get_terminal_size((120, 20)).columns - 1)
                rendered = line[:width]
                padding = " " * max(0, self.last_width - len(rendered))
                self.stream.write(f"\r{rendered}{padding}")
                self.stream.flush()
                self.last_width = len(rendered)
            else:
                print(line, file=self.stream)
            self.last_line = line

    def close(self) -> None:
        if self.overwrite and self.last_line:
            self.stream.write("\n")
            self.stream.flush()
        return


class TqdmProgressRenderer:
    def __init__(self, label: str) -> None:
        self.label = label
        self.bar = None
        self.phase = ""
        self.total: int | None = None
        self.plain = PlainProgressRenderer(label) if tqdm is None else None

    def __call__(self, update) -> None:
        total = update.total if update.total > 0 else None
        if tqdm is None:
            if self.plain is not None:
                self.plain(update)
            return
        if self.bar is None or self.phase != update.phase or self.total != total:
            self.close()
            self.phase = update.phase
            self.total = total
            self.bar = tqdm(
                total=total,
                desc=update.message,
                unit="step",
                dynamic_ncols=True,
                leave=False,
                mininterval=0.10,
            )
        self.bar.set_description_str(update.message)
        detail = format_progress_detail(update)
        if detail:
            self.bar.set_postfix_str(detail[:180])
        if total is None:
            self.bar.update(1)
        else:
            target = max(0, min(update.current, total))
            if target >= self.bar.n:
                self.bar.update(target - self.bar.n)
            else:
                self.bar.n = target
                self.bar.refresh()

    def close(self) -> None:
        if self.plain is not None:
            self.plain.close()
        if self.bar is not None:
            self.bar.close()
            self.bar = None


def format_progress_detail(update) -> str:
    parts: list[str] = []
    if update.detail:
        parts.append(str(update.detail))
    timing = []
    if update.elapsed_seconds is not None:
        timing.append(f"elapsed {format_duration(update.elapsed_seconds)}")
    if update.step_seconds is not None:
        timing.append(f"step {format_duration(update.step_seconds)}")
    if update.eta_seconds is not None:
        timing.append(f"eta {format_duration(update.eta_seconds)}")
    if timing:
        parts.append(", ".join(timing))
    return " | ".join(parts)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


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
