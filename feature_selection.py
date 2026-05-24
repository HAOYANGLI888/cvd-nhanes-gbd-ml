"""Compatibility wrapper for scripts/feature_selection.py."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "feature_selection.py"), run_name="__main__")
