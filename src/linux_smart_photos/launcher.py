from __future__ import annotations

import sys


CLI_COMMANDS = {"status", "sync", "search", "models"}


def print_usage() -> None:
    print(
        "\n".join(
            [
                "Usage: smart-photos [--gui] [--cli <command...>]",
                "",
                "Examples:",
                "  smart-photos",
                "  smart-photos --cli status",
                "  smart-photos --cli sync",
                "  smart-photos --cli search cat --limit 10",
            ]
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] in (["--help"], ["-h"]):
        print_usage()
        return 0

    if args[:1] in (["--gui"], ["gui"]):
        from .app import main as gui_main

        return gui_main()

    if args[:1] in (["--cli"], ["cli"]):
        from .cli import main as cli_main

        return cli_main(args[1:])

    if args and (args[0] in CLI_COMMANDS or args[0] == "--config"):
        from .cli import main as cli_main

        return cli_main(args)

    from .app import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
