import json
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model.network import FakeImageDetectorCNN
from utils.image_preprocess import collect_dataset_files, ensure_directories, load_image_as_array


MODEL_DIR = BASE_DIR / "model"
GENERATED_DIR = BASE_DIR / "static" / "generated"
MODEL_PATH = MODEL_DIR / "fake_image_detector.pth"
BATCH_SIZE = 32
EPOCHS = 12
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 25
PATIENCE = 4


class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image = load_image_as_array(self.file_paths[index])
        image = (image * 255).astype(np.uint8)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        label = torch.tensor([self.labels[index]], dtype=torch.float32)
        return image, label


def set_seed(seed_value=SEED):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def save_history_plots(history):
    epochs = range(1, len(history["train_accuracy"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy", linewidth=2)
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(GENERATED_DIR / "loss_accuracy_graph.png", dpi=200)
    plt.close()


def save_confusion_matrix_plot(matrix):
    plt.figure(figsize=(5, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_labels = ["Real", "Fake"]
    plt.xticks([0, 1], tick_labels)
    plt.yticks([0, 1], tick_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for row in range(2):
        for col in range(2):
            plt.text(col, row, matrix[row, col], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(GENERATED_DIR / "confusion_matrix.png", dpi=200)
    plt.close()


def evaluate_model(model, dataloader, criterion):
    model.eval()
    losses = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(loss.item())

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).int()
            y_true.extend(labels.cpu().numpy().astype(int).flatten().tolist())
            y_pred.extend(predictions.cpu().numpy().astype(int).flatten().tolist())

    accuracy = float(accuracy_score(y_true, y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return accuracy, matrix, avg_loss


def train_epoch(model, dataloader, criterion, optimizer, epoch_index):
    model.train()
    losses = []
    y_true = []
    y_pred = []
    total_batches = len(dataloader)

    for batch_index, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).int()
        y_true.extend(labels.detach().cpu().numpy().astype(int).flatten().tolist())
        y_pred.extend(predictions.detach().cpu().numpy().astype(int).flatten().tolist())

        if batch_index == 1 or batch_index % LOG_INTERVAL == 0 or batch_index == total_batches:
            print(
                f"Epoch {epoch_index} | Batch {batch_index}/{total_batches} | Loss: {loss.item():.4f}",
                flush=True,
            )

    avg_loss = float(np.mean(losses)) if losses else 0.0
    accuracy = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    return avg_loss, accuracy


def build_pos_weight(labels):
    labels_array = np.array(labels)
    positive_count = max(int(labels_array.sum()), 1)
    negative_count = max(int((labels_array == 0).sum()), 1)
    return torch.tensor([negative_count / positive_count], dtype=torch.float32, device=DEVICE)


def main():
    set_seed()
    ensure_directories([MODEL_DIR, GENERATED_DIR])

    print(f"Using device: {DEVICE}", flush=True)
    print("Scanning dataset folders...", flush=True)
    files, labels, dataset_note = collect_dataset_files(BASE_DIR)
    print(dataset_note, flush=True)

    if len(files) < 20:
        raise RuntimeError("Not enough images were found to train the model.")

    train_files, val_files, train_labels, val_labels = train_test_split(
        files,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    print(f"Training samples: {len(train_files)}", flush=True)
    print(f"Validation samples: {len(val_files)}", flush=True)
    print(f"Batch size: {BATCH_SIZE}", flush=True)
    print(f"Epochs: {EPOCHS}", flush=True)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop((128, 128), scale=(0.88, 1.0)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    train_dataset = ImageDataset(train_files, train_labels, transform=train_transform)
    val_dataset = ImageDataset(val_files, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FakeImageDetectorCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=build_pos_weight(train_labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    history = {"train_accuracy": [], "val_accuracy": [], "train_loss": [], "val_loss": []}

    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch + 1}/{EPOCHS}...", flush=True)
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, epoch + 1)
        val_accuracy, matrix, val_loss = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_accuracy)

        history["train_accuracy"].append(round(train_accuracy, 4))
        history["val_accuracy"].append(round(val_accuracy, 4))
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_accuracy,
                },
                MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{EPOCHS} complete | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | LR: {current_lr:.6f}",
            flush=True,
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.", flush=True)
            break

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_accuracy, matrix, validation_loss = evaluate_model(model, val_loader, criterion)

    save_history_plots(history)
    save_confusion_matrix_plot(np.array(matrix))

    metrics_payload = {
        "accuracy": round(validation_accuracy * 100, 2),
        "loss": round(float(validation_loss), 4),
        "confusion_matrix": matrix,
        "history": history,
        "dataset_note": dataset_note,
        "training_samples": len(train_files),
        "validation_samples": len(val_files),
        "loss_graph": "generated/loss_accuracy_graph.png",
        "confusion_matrix_image": "generated/confusion_matrix.png",
        "device": str(DEVICE),
        "best_val_accuracy": round(best_val_accuracy * 100, 2),
    }

    with (GENERATED_DIR / "training_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    print("Training complete.", flush=True)
    print(f"Model saved to: {MODEL_PATH}", flush=True)
    print(f"Validation accuracy: {metrics_payload['accuracy']}%", flush=True)
    print(f"Dataset note: {dataset_note}", flush=True)


if __name__ == "__main__":
    main()
