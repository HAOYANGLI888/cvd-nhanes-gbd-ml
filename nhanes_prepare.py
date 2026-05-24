"""Compatibility wrapper for scripts/nhanes_prepare.py."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "nhanes_prepare.py"), run_name="__main__")
