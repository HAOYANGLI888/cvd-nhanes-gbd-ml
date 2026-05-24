"""Compatibility wrapper for scripts/gbd_analysis.py."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "gbd_analysis.py"), run_name="__main__")
