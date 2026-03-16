import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = PROJECT_ROOT / "fake_image_detector"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app import app
