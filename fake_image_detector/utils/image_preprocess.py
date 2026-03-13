from pathlib import Path

import cv2
import numpy as np
import torch


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
IMAGE_SIZE = (128, 128)


def ensure_directories(paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image_as_array(image_path, target_size=IMAGE_SIZE):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    return image


def preprocess_single_image(image_path, target_size=IMAGE_SIZE, as_tensor=False):
    image = load_image_as_array(image_path, target_size=target_size)
    if as_tensor:
        tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        return tensor
    return np.expand_dims(image, axis=0)


def find_labeled_directories(base_dir):
    search_roots = [
        base_dir / "dataset",
        base_dir / "Dataset",
        base_dir.parent / "dataset",
        base_dir.parent / "Dataset",
    ]
    labeled_dirs = {"real": [], "fake": []}
    seen_paths = set()

    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            lower_name = path.name.lower()
            resolved = str(path.resolve())
            if resolved in seen_paths:
                continue
            if lower_name == "real":
                labeled_dirs["real"].append(path)
                seen_paths.add(resolved)
            elif lower_name == "fake":
                labeled_dirs["fake"].append(path)
                seen_paths.add(resolved)
    return labeled_dirs


def collect_dataset_files(base_dir):
    labeled_dirs = find_labeled_directories(base_dir)
    real_files = []
    fake_files = []

    for real_dir in labeled_dirs["real"]:
        real_files.extend(sorted([path for path in real_dir.iterdir() if path.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS]))
    for fake_dir in labeled_dirs["fake"]:
        fake_files.extend(sorted([path for path in fake_dir.iterdir() if path.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS]))

    files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)

    dataset_note = (
        f"Discovered {len(real_files)} real images and {len(fake_files)} fake images "
        f"from {len(labeled_dirs['real'])} real folders and {len(labeled_dirs['fake'])} fake folders."
    )
    return files, labels, dataset_note
