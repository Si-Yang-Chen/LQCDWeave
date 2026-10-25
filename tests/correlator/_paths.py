"""Stable locations for module tests after directory separation."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = PROJECT_ROOT / "tests"
DATA_ROOT = TESTS_ROOT / "data"
