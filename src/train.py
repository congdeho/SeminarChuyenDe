"""Phase 3 training. Run from project root: python -m src.train --run-dir runs/baseline_v1"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import DEFAULT_MANIFEST, build_train_val_loaders, train_class_weights
from src.labels import LABELS
from src.model import (ARCHITECTURE, ARCHITECTURES, PRETRAINED_WEIGHTS, build_model, enter_train_mode,
                       preprocessing_metadata, set_trainable_stage)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/train.yaml"
HISTORY_FIELDS = ("stage", "stage_epoch", "global_epoch", "train_loss", "train_accuracy",
                  "val_loss", "val_accuracy", "val_macro_f1", "learning_rate")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng(train_loader) -> dict:
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (np_state[0], np_state[1].tolist(),
                                                      np_state[2], np_state[3], np_state[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": train_loader.generator.get_state()}


def restore_rng(state: dict, train_loader) -> None:
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    train_loader.generator.set_state(state["loader"])


def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_loss += criterion(logits, targets).item() * len(targets)
            all_true.extend(targets.cpu().tolist())
            all_pred.extend(logits.argmax(1).cpu().tolist())
    return {"loss": total_loss / len(all_true),
            "accuracy": sum(a == b for a, b in zip(all_true, all_pred)) / len(all_true),
            "macro_f1": float(f1_score(all_true, all_pred, labels=list(range(len(LABELS))),
                                        average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(all_true, all_pred,
                                                  labels=list(range(len(LABELS)))).tolist(),
            "classification_report": classification_report(all_true, all_pred,
                                                             labels=list(range(len(LABELS))),
                                                             target_names=list(LABELS),
                                                             output_dict=True, zero_division=0)}


def train_one_epoch(model: nn.Module, loader, criterion: nn.Module, optimizer: AdamW,
                    device: torch.device, stage: str, max_batches: int | None = None) -> dict:
    enter_train_mode(model, stage)
    total_loss = total_correct = total_examples = 0
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at batch {batch_index}")
        loss.backward()
        optimizer.step()
        total_examples += len(targets)
        total_loss += loss.item() * len(targets)
        total_correct += (logits.argmax(1) == targets).sum().item()
    if not total_examples:
        raise ValueError("No training batches")
    return {"loss": total_loss / total_examples, "accuracy": total_correct / total_examples}


def save_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_history(history: list[dict], run_dir: Path) -> None:
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    epochs = [row["global_epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Learning curves")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Validation macro-F1", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(run_dir / "learning_curves.png", dpi=150)
    plt.close(fig)


def checkpoint_payload(model: nn.Module, config: dict, manifest_sha: str, history: list[dict],
                       best_score: float, stage: str, stage_epoch: int,
                       optimizer: AdamW | None, scheduler: ReduceLROnPlateau | None,
                       stage_best: float, stale_epochs: int, train_loader, smoke: bool) -> dict:
    return {"schema_version": 1, "architecture": config.get("architecture", ARCHITECTURE),
            "pretrained_weights": config["pretrained_weights"],
            "labels": list(LABELS), "preprocessing": preprocessing_metadata(config["image_size"]),
            "config": config, "manifest_sha256": manifest_sha, "seed": config["seed"],
            "versions": {"python": str(sys.version.split()[0]), "torch": str(torch.__version__),
                         "torchvision": str(torchvision.__version__)},
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "history": history, "best_val_macro_f1": best_score, "stage": stage,
            "stage_epoch": stage_epoch, "stage_best": stage_best, "stale_epochs": stale_epochs,
            "rng_state": capture_rng(train_loader), "smoke": smoke}


def run_training(config: dict, run_dir: Path, manifest_path: Path = DEFAULT_MANIFEST,
                 resume: bool = False, max_train_batches: int | None = None,
                 pretrained: bool = True, stop_after_epochs: int | None = None) -> dict:
    architecture = config.get("architecture", ARCHITECTURE)
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    expected_weights = "NONE" if architecture == "cnn_baseline" else PRETRAINED_WEIGHTS
    if config["pretrained_weights"] != expected_weights:
        raise ValueError(f"{architecture} requires pretrained_weights={expected_weights}")
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if not resume and (last_path.exists() or best_path.exists()):
        raise FileExistsError(f"Run directory has checkpoints; use --resume or a new --run-dir: {run_dir}")
    if resume and not last_path.exists():
        raise FileNotFoundError(f"No last.pt to resume: {last_path}")
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader = build_train_val_loaders(manifest_path, config["batch_size"],
                                                      config["num_workers"], config["seed"],
                                                      config["image_size"],
                                                      config.get("augmentation", True))
    manifest_sha = file_sha256(manifest_path)
    weights = train_class_weights(manifest_path).to(device) if config["use_class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    saved = torch.load(last_path, map_location="cpu", weights_only=True) if resume else None
    if saved:
        if saved["manifest_sha256"] != manifest_sha or saved["config"] != config:
            raise ValueError("Resume requires the same manifest and training configuration")
    model = build_model(pretrained=pretrained and not resume and architecture != "cnn_baseline",
                        architecture=architecture).to(device)
    if saved:
        model.load_state_dict(saved["model_state"])
    history = list(saved["history"]) if saved else []
    best_score = float(saved["best_val_macro_f1"]) if saved else -1.0
    resume_stage = saved["stage"] if saved else "classifier"
    smoke = max_train_batches is not None
    if saved and saved["smoke"] != smoke:
        raise ValueError("Cannot resume a smoke run as a full run (or vice versa)")
    stages = [("classifier", config["classifier_epochs"], config["classifier_lr"]),
              ("finetune", config["finetune_epochs"], config["finetune_lr"])]
    completed = resume_stage == "complete"
    interrupted = False
    for stage, epoch_limit, lr in stages:
        if completed or (resume_stage == "finetune" and stage == "classifier") or epoch_limit == 0:
            continue
        set_trainable_stage(model, stage)
        optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr,
                          weight_decay=config["weight_decay"])
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=config["scheduler_factor"],
                                     patience=config["scheduler_patience"])
        if saved and resume_stage == stage:
            if saved["optimizer_state"] is not None:
                optimizer.load_state_dict(saved["optimizer_state"])
                scheduler.load_state_dict(saved["scheduler_state"])
            stage_start = saved["stage_epoch"] + 1
            stage_best = float(saved["stage_best"])
            stale_epochs = int(saved["stale_epochs"])
            restore_rng(saved["rng_state"], train_loader)
        else:
            stage_start, stage_best, stale_epochs = 0, -1.0, 0
        for stage_epoch in range(stage_start, epoch_limit):
            if stale_epochs >= config["early_stopping_patience"]:
                break
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                            stage, max_train_batches)
            val_metrics = evaluate(model, val_loader, criterion, device)
            score = val_metrics["macro_f1"]
            row = {"stage": stage, "stage_epoch": stage_epoch + 1,
                   "global_epoch": len(history) + 1,
                   "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
                   "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
                   "val_macro_f1": score, "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(row)
            if score > stage_best:
                stage_best, stale_epochs = score, 0
            else:
                stale_epochs += 1
            scheduler.step(score)
            improved = score > best_score
            if improved:
                best_score = score
            payload = checkpoint_payload(model, config, manifest_sha, history, best_score,
                                         stage, stage_epoch, optimizer, scheduler,
                                         stage_best, stale_epochs, train_loader, smoke)
            save_atomic(payload, last_path)
            if improved:
                save_atomic(payload, best_path)
                (run_dir / "best_validation_metrics.json").write_text(
                    json.dumps({"stage": stage, "epoch": len(history), "labels": LABELS,
                                "validation": val_metrics}, indent=2) + "\n", encoding="utf-8")
            save_history(history, run_dir)
            print(f"{stage} {stage_epoch + 1}/{epoch_limit} | train_loss={row['train_loss']:.4f} "
                  f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f} "
                  f"val_macro_f1={score:.4f} best={best_score:.4f}", flush=True)
            if stop_after_epochs is not None and len(history) >= stop_after_epochs:
                interrupted = True
                break
            if stale_epochs >= config["early_stopping_patience"]:
                print(f"Early stopping {stage} after {stale_epochs} stale epochs", flush=True)
                break
        if interrupted:
            break
        # A resume after the boundary must not re-run the prior stage.
        saved = None
        resume_stage = "finetune"
    if history and not interrupted:
        latest = torch.load(last_path, map_location="cpu", weights_only=True)
        latest["stage"] = "complete"
        latest["optimizer_state"] = None
        latest["scheduler_state"] = None
        save_atomic(latest, last_path)
    summary = {"device": str(device), "architecture": architecture,
               "augmentation": config.get("augmentation", True), "seed": config["seed"],
               "epochs_run": len(history),
               "best_val_macro_f1": best_score, "manifest_sha256": manifest_sha,
               "best_checkpoint": str(best_path), "smoke": smoke,
               "test_split_used": False, "complete": not interrupted}
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026), default=None,
                        help="Override the planned experimental seed; use the same value on --resume")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=None,
                        help="Limit train batches per epoch; checkpoint is marked smoke")
    parser.add_argument("--stop-after-epochs", type=int, default=None,
                        help="Exit cleanly after this many total epochs, then use --resume")
    parser.add_argument("--no-pretrained", action="store_true", help="Only for offline smoke tests")
    args = parser.parse_args()
    if args.no_pretrained and args.smoke_batches is None:
        parser.error("--no-pretrained is allowed only for smoke runs")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    result = run_training(config, args.run_dir, args.manifest, args.resume,
                          args.smoke_batches, pretrained=not args.no_pretrained,
                          stop_after_epochs=args.stop_after_epochs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
