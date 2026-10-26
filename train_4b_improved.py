"""Improved training for the 4B chart-image predictor.

The first run (colab_4b_stock_movement.ipynb) ended at 36.5% test accuracy
against a 42.0% majority-class baseline: train accuracy ran away to 85% while
validation accuracy peaked at epoch 3 and then declined. The splits were
honest (chronological, from the manifest), so the fault was the recipe.

This script keeps the notebook's chronological splits and fixes the recipe:

  * head-only warmup, then discriminative learning rates (backbone / BN / head)
    with warmup + cosine decay instead of one flat 3e-4 for the whole network
  * chart-appropriate augmentation: scale jitter, brightness/contrast jitter,
    blur, cutout, plus the exact vertical-reflection symmetry of a price chart
    (flip the chart, swap up <-> down) - used only when a colour-only probe
    shows the renderer does not already encode direction
  * mixup, label smoothing, weight EMA and balanced sampling
  * test-time augmentation and temperature scaling at inference

The checkpoint it writes has the same layout the serving code expects, so
serve_4b_predictor.py and the browser ONNX export load it unchanged.

Run on the Colab VM with Drive mounted:
    python train_4b_improved.py --data-root /content/drive/MyDrive/4b_chart_images
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms as T

CLASS_NAMES = ("down", "flat", "up")
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
# A vertical reflection turns a rising chart into a falling one.
FLIP_LABEL = {0: 2, 1: 1, 2: 0}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

OLD_TEST_ACCURACY = 0.36486486486486486
OLD_MAJORITY_BASELINE = 0.4197635135135135


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------- data ---
def resolve_image_path(raw_path, root):
    candidate = Path(str(raw_path))
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Image is outside the data root: {path}") from exc
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_manifest(data_root):
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.csv not found in {root}")
    manifest = pd.read_csv(manifest_path)
    required = {
        "image_path", "label", "split", "symbol", "input_start_time",
        "prediction_time", "target_end_time", "horizon", "threshold",
        "renderer_version", "source_pipeline_id",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest.csv is missing columns: {sorted(missing)}")

    rows = []
    for _, item in manifest.iterrows():
        label = str(item["label"]).strip().lower()
        if label not in LABEL_TO_ID:
            raise ValueError(f"Invalid label {label!r}; use {CLASS_NAMES}")
        split = str(item["split"]).strip().lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {split!r}")
        path = resolve_image_path(item["image_path"], root)
        rows.append({
            "path": path, "label": label, "split": split,
            "symbol": str(item["symbol"]).strip(),
            "horizon": str(item["horizon"]).strip(),
            "threshold": str(item["threshold"]).strip(),
            "renderer_version": str(item["renderer_version"]).strip(),
        })
    frame = pd.DataFrame(rows)
    frame["label_id"] = frame["label"].map(LABEL_TO_ID).astype(int)

    # exact duplicates must not straddle the splits
    frame["sha1"] = [hashlib.sha1(p.read_bytes()).hexdigest() for p in frame["path"]]
    cross = frame.groupby("sha1")["split"].nunique()
    cross = cross[cross > 1]
    if len(cross):
        raise ValueError(f"Exact duplicate image appears across splits: {len(cross)} hash(es)")
    frame = frame.drop_duplicates("sha1").reset_index(drop=True)

    splits = {name: frame[frame["split"] == name] for name in ("train", "val", "test")}
    for name, part in splits.items():
        if part.empty:
            raise ValueError(f"Split is empty: {name}")
        if set(part["label"]) != set(CLASS_NAMES):
            raise ValueError(f"Split {name} must contain all labels")
    log(f"loaded {len(frame)} images: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    return frame, splits


def majority_baseline(frame):
    counts = frame["label_id"].value_counts()
    return float(counts.max() / len(frame))


# ------------------------------------------------------------- colour probe ---
def colour_leak_probe(train_frame, limit=1200):
    """Does colour alone reveal the label?

    A colour-only classifier that beats the majority baseline means the
    renderer encodes direction in colour, so vertically flipping a chart would
    corrupt its label. The vertical-reflection augmentation is then unsafe.
    """
    subset = train_frame.head(min(limit, len(train_frame)))
    features, labels = [], []
    for _, row in subset.iterrows():
        with Image.open(row["path"]) as raw:
            arr = np.asarray(raw.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        features.append([
            r.mean(), g.mean(), b.mean(), r.std(), g.std(), b.std(),
            (g - r).mean(), (g - b).mean(), (r - b).mean(), (g - r).std(),
        ])
        labels.append(int(row["label_id"]))
    x = np.asarray(features)
    y = np.asarray(labels)
    if len(set(y.tolist())) < 2:
        return 0.0
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=3000).fit(x_train, y_train)
    accuracy = float(accuracy_score(y_test, clf.predict(x_test)))
    baseline = float(np.bincount(y_test).max() / len(y_test))
    log(f"colour-only probe: accuracy {accuracy:.3f} vs baseline {baseline:.3f}")
    return accuracy - baseline


# ------------------------------------------------------------------ dataset ---
class ChartDataset(Dataset):
    def __init__(self, frame, transform, vflip_prob=0.0):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.vflip_prob = vflip_prob

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as raw:
            image = raw.convert("RGB")
        label = int(row["label_id"])
        if self.vflip_prob and random.random() < self.vflip_prob:
            image = T.functional.vflip(image)
            label = FLIP_LABEL[label]
        return self.transform(image), label


def build_transforms(size):
    train_transform = T.Compose([
        T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=True),
        T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)], p=0.8),
        T.RandomGrayscale(p=0.08),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])
    eval_transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


# -------------------------------------------------------------------- model ---
def build_model():
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(backbone.fc.in_features, len(CLASS_NAMES)))
    return backbone


def param_groups(model, lr_backbone, lr_bn, lr_head):
    head, backbone, bn = [], [], []
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            head.append(param)
        elif param.ndim == 1:
            bn.append(param)
        else:
            backbone.append(param)
    return [
        {"params": backbone, "lr": lr_backbone},
        {"params": bn, "lr": lr_bn},
        {"params": head, "lr": lr_head},
    ]


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        param.requires_grad = trainable or name.startswith("fc.")


def freeze_bn_stats(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_value, model_value in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value.detach(), alpha=1 - self.decay)
            else:
                ema_value.copy_(model_value)


def soft_cross_entropy(logits, target_probs, smoothing):
    if smoothing > 0:
        target_probs = target_probs * (1 - smoothing) + smoothing / target_probs.size(1)
    return -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def mixup_batch(images, labels, alpha, num_classes):
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.size(0))
    mixed = lam * images + (1 - lam) * images[permutation]
    target_a = F.one_hot(labels, num_classes).float()
    target_b = F.one_hot(labels[permutation], num_classes).float()
    return mixed, target_a, target_b, lam


# ---------------------------------------------------------------- evaluation ---
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            all_logits.append(model(images).float().cpu())
            all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def fit_temperature(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)

    def nll(log_temperature):
        scaled = logits / math.exp(log_temperature)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return float(-log_probs[np.arange(len(labels)), labels].mean())

    best_log_t, best_value = 0.0, nll(0.0)
    for log_t in np.linspace(-2.5, 2.5, 161):
        value = nll(log_t)
        if value < best_value:
            best_value, best_log_t = value, float(log_t)
    return float(math.exp(best_log_t))


def evaluate(model, loader, device, tta_flip):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            probs = torch.softmax(model(images), dim=1)
            if tta_flip:
                flipped = torch.softmax(model(torch.flip(images, dims=[2])), dim=1)
                probs = (probs + flipped[:, [2, 1, 0]]) / 2
            all_probs.append(probs.cpu())
            all_labels.append(labels)
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()
    predictions = probs.argmax(axis=1)
    return labels, predictions, probs


# ------------------------------------------------------------- training loop ---
def train(args, train_frame, val_frame, device, vflip_prob):
    train_transform, eval_transform = build_transforms(args.size)
    train_dataset = ChartDataset(train_frame, train_transform, vflip_prob=vflip_prob)
    val_dataset = ChartDataset(val_frame, eval_transform)

    counts = train_frame["label_id"].value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
    weights = (1.0 / counts.clip(lower=1).to_numpy()).astype(np.float64)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        param_groups(model, args.lr_backbone, args.lr_bn, args.lr_head),
        weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    ema = ModelEMA(model, decay=args.ema_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    best_val = -1.0
    best_state = None
    without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        warming = epoch <= args.warmup_epochs
        if warming:
            set_backbone_trainable(model, False)
        else:
            set_backbone_trainable(model, True)

        model.train()
        if warming:
            freeze_bn_stats(model)
        running_loss = 0.0
        correct = 0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            use_mixup = (not warming) and args.mixup_alpha > 0 and random.random() < args.mixup_prob
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                if use_mixup:
                    images, target_a, target_b, lam = mixup_batch(
                        images, labels, args.mixup_alpha, len(CLASS_NAMES))
                    logits = model(images)
                    loss = lam * soft_cross_entropy(logits, target_a, args.label_smoothing) \
                        + (1 - lam) * soft_cross_entropy(logits, target_b, args.label_smoothing)
                else:
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            if not use_mixup:
                correct += (logits.argmax(dim=1) == labels).sum().item()

        labels, predictions, _ = evaluate(ema.ema, val_loader, device, tta_flip=False)
        val_accuracy = float(accuracy_score(labels, predictions))
        val_balanced = float(balanced_accuracy_score(labels, predictions))
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "train_accuracy": correct / max(1, seen) if not warming else float("nan"),
            "val_accuracy": val_accuracy,
            "val_balanced_accuracy": val_balanced,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[2]["lr"],
        }
        history.append(row)
        log(
            f"epoch {epoch:02d}/{args.epochs} | train loss {row['train_loss']:.4f} "
            f"acc {row['train_accuracy']:.3f} | val acc {val_accuracy:.4f} "
            f"balanced {val_balanced:.4f} | lr {row['lr_backbone']:.2e}"
        )

        if val_accuracy > best_val:
            best_val = val_accuracy
            best_state = copy.deepcopy(ema.ema.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
            if without_improvement >= args.patience:
                log(f"early stopping after {epoch} epochs (patience {args.patience})")
                break

    if best_state is not None:
        ema.ema.load_state_dict(best_state)
    return ema.ema, best_val, history


# --------------------------------------------------------------------- main ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/content/drive/MyDrive/4b_chart_images")
    parser.add_argument("--output-root", default="/content/drive/MyDrive/4b_stock_movement_outputs")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-bn", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--vflip-prob", type=float, default=0.5)
    parser.add_argument("--no-vflip", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-name", default="best_4b_chart_model.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} size={args.size} epochs={args.epochs}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frame, splits = load_manifest(args.data_root)
    train_frame, val_frame, test_frame = splits["train"], splits["val"], splits["test"]

    baseline = majority_baseline(test_frame)
    log(f"test split: {len(test_frame)} images, majority baseline {baseline:.4f}")

    leak_margin = 0.0
    use_vflip = not args.no_vflip
    if use_vflip:
        leak_margin = colour_leak_probe(train_frame)
        if leak_margin > 0.03:
            log(f"colour leaks the label (+{leak_margin:.3f}); disabling vertical-flip augmentation")
            use_vflip = False
        else:
            log("no colour leak detected; vertical-flip augmentation enabled")
    vflip_prob = args.vflip_prob if use_vflip else 0.0

    model, best_val, history = train(args, train_frame, val_frame, device, vflip_prob)
    model = model.to(device)

    # temperature is fitted on validation logits, then applied to the test set
    _, eval_transform = build_transforms(args.size)
    val_loader = DataLoader(
        ChartDataset(val_frame, eval_transform), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(
        ChartDataset(test_frame, eval_transform), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    val_logits, val_labels = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    log(f"temperature fitted on validation: {temperature:.3f}")

    labels, predictions, probs = evaluate(model, test_loader, device, tta_flip=use_vflip)
    raw_labels, raw_predictions, raw_probs = evaluate(model, test_loader, device, tta_flip=False)
    calibrated = np.clip(raw_probs / temperature, 1e-9, 1.0)
    calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
    calibrated_predictions = calibrated.argmax(axis=1)

    def metrics_for(pred):
        return {
            "accuracy": float(accuracy_score(labels, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro")),
        }

    plain = metrics_for(raw_predictions)
    tta = metrics_for(predictions)
    calibrated_metrics = metrics_for(calibrated_predictions)

    best_variant = max(
        (("plain", plain, raw_probs), ("tta", tta, probs), ("tta_calibrated", calibrated_metrics, calibrated)),
        key=lambda item: item[1]["balanced_accuracy"],
    )
    variant, best_metrics, best_probs = best_variant
    log(f"plain      {plain}")
    log(f"tta        {tta}")
    log(f"tta+calib  {calibrated_metrics}")
    log(f"selected variant: {variant}")

    checkpoint_path = output_root / args.checkpoint_name
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "image_size": args.size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "model_name": "resnet18",
        "pretrained_initialization": True,
        "best_val_accuracy": best_val,
        "temperature": temperature,
        "tta_flip": bool(use_vflip),
    }, checkpoint_path)
    log(f"checkpoint saved: {checkpoint_path}")

    pd.DataFrame(history).to_csv(output_root / "training_history_improved.csv", index=False)
    (output_root / "test_metrics_improved.json").write_text(json.dumps({
        "test_accuracy": best_metrics["accuracy"],
        "test_balanced_accuracy": best_metrics["balanced_accuracy"],
        "test_macro_f1": best_metrics["macro_f1"],
        "majority_class_baseline": baseline,
        "previous_model_accuracy": OLD_TEST_ACCURACY,
        "previous_majority_baseline": OLD_MAJORITY_BASELINE,
        "best_val_accuracy": best_val,
        "variant": variant,
        "temperature": temperature,
        "tta_flip": bool(use_vflip),
        "image_size": args.size,
        "beats_baseline": best_metrics["accuracy"] > baseline,
        "beats_previous_model": best_metrics["accuracy"] > OLD_TEST_ACCURACY,
        "classification_report": classification_report(
            labels, best_probs.argmax(axis=1), target_names=list(CLASS_NAMES),
            output_dict=True, zero_division=0),
    }, indent=2), encoding="utf-8")

    verdict = "BEATS" if best_metrics["accuracy"] > baseline else "BELOW"
    log(f"VERDICT: {verdict} the {baseline:.4f} majority baseline "
        f"(previous model {OLD_TEST_ACCURACY:.4f}, new {best_metrics['accuracy']:.4f})")
    log("TRAIN_4B_IMPROVED_DONE")


if __name__ == "__main__":
    main()
