"""Frozen-bundle entry point for the S-Stream headless solver.

Runs the CLI; if startup fails, writes a traceback to the temp dir and (on
Windows) shows it in a message box, so a windowed build never dies silently.
"""

from __future__ import annotations

import sys
import traceback


def main() -> int:
    try:
        from sstream.cli import main as run

        return run()
    except Exception:
        tb = traceback.format_exc()
        try:
            import os
            import tempfile

            log_path = os.path.join(tempfile.gettempdir(), "sstream_error.log")
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(tb)
        except Exception:
            log_path = "(could not write log)"
        if sys.platform.startswith("win"):
            try:
                import ctypes

                snippet = tb if len(tb) < 1200 else tb[-1200:]
                ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
                    None,
                    f"S-Stream failed to start.\n\n{snippet}\n\nFull log: {log_path}",
                    "S-Stream — startup error",
                    0x10,  # MB_ICONERROR
                )
            except Exception:
                pass
        else:
            print(tb, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
