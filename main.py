from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR / "fake_image_detector"

# Make `fake_image_detector/app.py` importable as the top-level module `app`,
# which also allows its existing `from utils...` imports to work unchanged.
sys.path.insert(0, str(APP_DIR))

from app import app as flask_app  # noqa: E402

