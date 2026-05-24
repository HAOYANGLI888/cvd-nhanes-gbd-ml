"""Compatibility wrapper for scripts/ml_modeling.py."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "ml_modeling.py"), run_name="__main__")
