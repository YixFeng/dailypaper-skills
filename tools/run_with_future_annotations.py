#!/usr/bin/env python3
"""Execute a Python script after prepending postponed annotation evaluation.

This keeps Python 3.9 compatible with scripts that use PEP 604 unions like
`str | None` in annotations while preserving stdin/stdout behavior.
"""

from __future__ import annotations

import pathlib
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_with_future_annotations.py SCRIPT_PATH [ARGS...]", file=sys.stderr)
        return 2

    script_path = pathlib.Path(sys.argv[1]).resolve()
    script_args = sys.argv[1:]

    with script_path.open("r", encoding="utf-8") as f:
        source = f.read()

    code = compile("from __future__ import annotations\n" + source, str(script_path), "exec")
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(script_path),
        "__package__": None,
    }

    old_argv = sys.argv
    try:
        sys.argv = script_args
        exec(code, globals_dict)
    finally:
        sys.argv = old_argv

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
