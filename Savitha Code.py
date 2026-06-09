import os
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    PrecisionRecallDisplay,
)


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Dataset and preprocessing
# -----------------------------

class OralCancerImageDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None, gaussian_kernel: int = 5):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.gaussian_kernel = gaussian_kernel

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Stage 1: Gaussian filtering noise removal
        if self.gaussian_kernel > 0:
            k = self.gaussian_kernel if self.gaussian_kernel % 2 == 1 else self.gaussian_kernel + 1
            image = cv2.GaussianBlur(image, (k, k), 0)

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def load_image_paths(data_dir: str) -> Tuple[List[str], List[int], Dict[int, str]]:
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if len(class_names) < 2:
        raise ValueError("Dataset directory must contain at least two class folders, e.g., Cancer and Non-Cancer.")

    label_map = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in label_map.items()}

    image_paths, labels = [], []
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        for root, _, files in os.walk(cls_dir):
            for f in files:
                if f.lower().endswith(valid_ext):
                    image_paths.append(os.path.join(root, f))
                    labels.append(label_map[cls])

    if len(image_paths) == 0:
        raise ValueError("No image files found in dataset directory.")
    return image_paths, labels, idx_to_class


# -----------------------------
# HybridNet-style feature extractor
# -----------------------------

class FeatureExtractor(nn.Module):
    """HybridNet-style deep feature extractor using pretrained AlexNet features."""
    def __init__(self, device: str = "cuda"):
        super().__init__()
        weights = models.AlexNet_Weights.IMAGENET1K_V1
        alexnet = models.alexnet(weights=weights)
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(*list(alexnet.classifier.children())[:-1])  # 4096-dim feature
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def extract_features(loader: DataLoader, extractor: nn.Module, device: str) -> Tuple[np.ndarray, np.ndarray]:
    feats, labs = [], []
    extractor.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            out = extractor(images).cpu().numpy()
            feats.append(out)
            labs.append(labels.numpy())
    return np.vstack(feats), np.concatenate(labs)


# -----------------------------
# DSSAE classifier
# -----------------------------

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class DSSAEClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    weight_decay: float,
    epochs: int,
    device: str,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses, train_preds, train_true = [], [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            train_true.extend(y.cpu().numpy())

        model.eval()
        val_losses, val_preds, val_true = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_true.extend(y.cpu().numpy())

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))
        history["train_acc"].append(accuracy_score(train_true, train_preds))
        history["val_acc"].append(accuracy_score(val_true, val_preds))

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {history['train_loss'][-1]:.4f} | Train Acc: {history['train_acc'][-1]:.4f} | "
            f"Val Loss: {history['val_loss'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f}"
        )
    return history


class FeatureTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# Chicken Swarm Optimization
# -----------------------------

@dataclass
class CSOConfig:
    population_size: int = 8
    iterations: int = 5
    rooster_ratio: float = 0.2
    hen_ratio: float = 0.6
    chick_ratio: float = 0.2
    seed: int = 42


class ChickenSwarmOptimizer:
    """
    Simplified CSO for tuning DSSAE hyperparameters.
    Search vector: [learning_rate, dropout, hidden1, hidden2, weight_decay]
    """
    def __init__(self, bounds: List[Tuple[float, float]], config: CSOConfig):
        self.bounds = np.array(bounds, dtype=float)
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def initialize(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return self.rng.uniform(low, high, size=(self.config.population_size, len(self.bounds)))

    def clip(self, pop):
        return np.clip(pop, self.bounds[:, 0], self.bounds[:, 1])

    @staticmethod
    def decode(position):
        lr = float(position[0])
        dropout = float(position[1])
        hidden1 = int(round(position[2] / 16) * 16)
        hidden2 = int(round(position[3] / 16) * 16)
        weight_decay = float(position[4])
        hidden1 = max(32, hidden1)
        hidden2 = max(16, hidden2)
        return {
            "lr": lr,
            "dropout": dropout,
            "hidden_dims": [hidden1, hidden2],
            "weight_decay": weight_decay,
        }

    def optimize(self, objective_fn):
        pop = self.initialize()
        fitness = np.array([objective_fn(self.decode(p)) for p in pop])
        best_idx = np.argmin(fitness)
        best_pos, best_fit = pop[best_idx].copy(), fitness[best_idx]

        n = len(pop)
        n_roosters = max(1, int(n * self.config.rooster_ratio))
        n_hens = max(1, int(n * self.config.hen_ratio))

        for it in range(self.config.iterations):
            order = np.argsort(fitness)
            roosters = order[:n_roosters]
            hens = order[n_roosters:n_roosters + n_hens]
            chicks = order[n_roosters + n_hens:]

            new_pop = pop.copy()

            # Rooster update: Gaussian perturbation, better roosters move less
            for idx in roosters:
                sigma = np.exp(-fitness[idx] / (np.abs(best_fit) + 1e-8))
                new_pop[idx] = pop[idx] * (1 + self.rng.normal(0, sigma, size=pop.shape[1]))

            # Hen update: move toward a rooster and away/toward another random chicken
            for idx in hens:
                r1 = self.rng.choice(roosters)
                r2 = self.rng.integers(0, n)
                s1 = np.exp((fitness[idx] - fitness[r1]) / (np.abs(fitness[idx]) + 1e-8))
                s2 = np.exp(fitness[r2] - fitness[idx])
                rand1 = self.rng.random(pop.shape[1])
                rand2 = self.rng.random(pop.shape[1])
                new_pop[idx] = pop[idx] + rand1 * s1 * (pop[r1] - pop[idx]) + rand2 * s2 * (pop[r2] - pop[idx])

            # Chick update: follow a mother hen
            for idx in chicks:
                mother = self.rng.choice(hens) if len(hens) else self.rng.integers(0, n)
                fl = self.rng.uniform(0.4, 0.9)
                new_pop[idx] = pop[idx] + fl * (pop[mother] - pop[idx])

            pop = self.clip(new_pop)
            fitness = np.array([objective_fn(self.decode(p)) for p in pop])
            current_best = np.argmin(fitness)
            if fitness[current_best] < best_fit:
                best_fit = fitness[current_best]
                best_pos = pop[current_best].copy()

            print(f"CSO Iteration {it+1}/{self.config.iterations} | Best objective: {best_fit:.4f}")

        return self.decode(best_pos), best_fit


# -----------------------------
# Evaluation and plotting
# -----------------------------

def evaluate_model(model, loader, device, class_names):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_prob.extend(probs[:, 1] if probs.shape[1] > 1 else probs[:, 0])
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    print("\nOverall Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v * 100:.2f}%" if k != "MCC" else f"{k}: {v:.4f}")

    return y_true, y_pred, y_prob, metrics


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_history(history, out_dir):
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_acc"], label="Training")
    plt.plot(history["val_acc"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="Training")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=300)
    plt.close()


def plot_precision_recall(y_true, y_prob, out_path):
    plt.figure(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------
# Main experiment
# -----------------------------

def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    image_paths, labels, idx_to_class = load_image_paths(args.data_dir)
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Total images: {len(image_paths)}")

    train_paths, test_paths, y_train, y_test = train_test_split(
        image_paths,
        labels,
        train_size=args.split_ratio,
        random_state=args.seed,
        stratify=labels,
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_img_ds = OralCancerImageDataset(train_paths, y_train, transform=transform, gaussian_kernel=args.gaussian_kernel)
    test_img_ds = OralCancerImageDataset(test_paths, y_test, transform=transform, gaussian_kernel=args.gaussian_kernel)
    train_img_loader = DataLoader(train_img_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_img_loader = DataLoader(test_img_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Extracting HybridNet-style deep features...")
    extractor = FeatureExtractor(device=device)
    X_train, y_train_np = extract_features(train_img_loader, extractor, device)
    X_test, y_test_np = extract_features(test_img_loader, extractor, device)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train_np,
        test_size=0.2,
        random_state=args.seed,
        stratify=y_train_np,
    )

    input_dim = X_train.shape[1]

    def objective(params):
        model = DSSAEClassifier(
            input_dim=input_dim,
            hidden_dims=params["hidden_dims"],
            num_classes=num_classes,
            dropout=params["dropout"],
        ).to(device)

        train_ds = FeatureTensorDataset(X_train_sub, y_train_sub)
        val_ds = FeatureTensorDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=args.feature_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.feature_batch_size, shuffle=False)

        hist = train_classifier(
            model,
            train_loader,
            val_loader,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            epochs=args.cso_eval_epochs,
            device=device,
        )
        return 1.0 - max(hist["val_acc"])

    if args.use_cso:
        bounds = [
            (1e-5, 5e-3),   # learning rate
            (0.1, 0.6),     # dropout
            (64, 1024),     # hidden layer 1
            (32, 512),      # hidden layer 2
            (1e-6, 1e-3),   # weight decay
        ]
        cso = ChickenSwarmOptimizer(bounds, CSOConfig(
            population_size=args.cso_population,
            iterations=args.cso_iterations,
            seed=args.seed,
        ))
        best_params, best_obj = cso.optimize(objective)
        print(f"Best CSO params: {best_params}; objective={best_obj:.4f}")
    else:
        best_params = {
            "lr": args.lr,
            "dropout": args.dropout,
            "hidden_dims": [args.hidden1, args.hidden2],
            "weight_decay": args.weight_decay,
        }

    # Final training on training split with validation taken from train
    final_model = DSSAEClassifier(
        input_dim=input_dim,
        hidden_dims=best_params["hidden_dims"],
        num_classes=num_classes,
        dropout=best_params["dropout"],
    ).to(device)

    train_ds = FeatureTensorDataset(X_train_sub, y_train_sub)
    val_ds = FeatureTensorDataset(X_val, y_val)
    test_ds = FeatureTensorDataset(X_test, y_test_np)

    train_loader = DataLoader(train_ds, batch_size=args.feature_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.feature_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.feature_batch_size, shuffle=False)

    print("\nTraining final DSSAE classifier...")
    history = train_classifier(
        final_model,
        train_loader,
        val_loader,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        epochs=args.epochs,
        device=device,
    )

    print("\nTesting final CSOHN-OCC model...")
    y_true, y_pred, y_prob, metrics = evaluate_model(final_model, test_loader, device, class_names)

    torch.save({
        "model_state_dict": final_model.state_dict(),
        "best_params": best_params,
        "class_names": class_names,
        "metrics": metrics,
    }, os.path.join(args.output_dir, "csohn_occ_model.pt"))

    plot_history(history, args.output_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(args.output_dir, "confusion_matrix.png"))
    if num_classes == 2:
        plot_precision_recall(y_true, y_prob, os.path.join(args.output_dir, "precision_recall_curve.png"))

    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write("CSOHN-OCC Overall Metrics\n")
        f.write(str(best_params) + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"\nSaved outputs to: {args.output_dir}")
    print("Generated files: model checkpoint, metrics.txt, confusion_matrix.png, accuracy_curve.png, loss_curve.png, PR curve.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSOHN-OCC Oral Cancer Classification Overall Code")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to oral cancer image dataset folder")
    parser.add_argument("--output_dir", type=str, default="outputs_csohn_occ")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Training split ratio, e.g., 0.8 for 80:20 or 0.7 for 70:30")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--feature_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gaussian_kernel", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # Default hyperparameters when CSO is disabled
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # CSO settings
    parser.add_argument("--use_cso", action="store_true", help="Enable CSO hyperparameter tuning")
    parser.add_argument("--cso_population", type=int, default=8)
    parser.add_argument("--cso_iterations", type=int, default=5)
    parser.add_argument("--cso_eval_epochs", type=int, default=5)

    args = parser.parse_args()
    main(args)
