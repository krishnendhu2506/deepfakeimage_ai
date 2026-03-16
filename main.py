import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "fake_image_detector"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
