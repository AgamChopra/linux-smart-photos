from __future__ import annotations

import os
import subprocess
import sys

from .branding import project_root

CLI_COMMANDS = {"status", "sync", "search", "models", "migrate"}


def print_usage() -> None:
    print(
        "\n".join(
            [
                "Usage: smart-photos [--gui|--electron|--qt] [--cli <command...>]",
                "",
                "Examples:",
                "  smart-photos",
                "  smart-photos --cli status",
                "  smart-photos --cli sync",
                "  smart-photos --cli search cat --limit 10",
                "  smart-photos --qt",
            ]
        )
    )


def _launch_electron(args: list[str]) -> int:
    root = project_root()
    electron_binary = root / "electron" / "node_modules" / ".bin" / "electron"
    if not electron_binary.exists():
        from .app import main as gui_main

        return gui_main()

    env = os.environ.copy()
    env.setdefault("SMART_PHOTOS_VENV_PYTHON", sys.executable)
    command = [str(electron_binary), str(root / "electron"), *args]
    return subprocess.call(command, cwd=str(root), env=env)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] in (["--help"], ["-h"]):
        print_usage()
        return 0

    if args[:1] in (["--gui"], ["gui"], ["--electron"], ["electron"]):
        return _launch_electron(args[1:])

    if args[:1] in (["--qt"], ["qt"]):
        from .app import main as gui_main

        return gui_main()

    if args[:1] in (["--cli"], ["cli"]):
        from .cli import main as cli_main

        return cli_main(args[1:])

    if args and (args[0] in CLI_COMMANDS or args[0] == "--config"):
        from .cli import main as cli_main

        return cli_main(args)

    return _launch_electron(args)


if __name__ == "__main__":
    raise SystemExit(main())
