"""Train a material specialist and select how to use it on validation only."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.camera import select_device
from src.dataset import DEFAULT_MANIFEST, ROOT, read_manifest
from src.evaluate import calculate_metrics, evaluate_split
from src.labels import LABELS
from src.material_model import (ARCHITECTURE, BOTTLE_ARCHITECTURE, GLASS, PEERS, PLASTIC, BottleModel, MaterialModel,
                                build_encoder, embeddings, full_view)
from src.predict import load_predictor
from src.select_threshold import sweep_validation
from src.train import file_sha256, save_atomic


RUNS = ROOT / "runs/phase5"
CHECKPOINTS = {"e2": RUNS / "e2_seed42/best.pt", "e1": RUNS / "e1_seed42/best.pt",
               "e3": ROOT / "runs/local_pretrained_probe/best.pt", "e4": RUNS / "e4_seed42/best.pt"}
DEFAULT_OUTPUT = ROOT / "runs/material_refinement"
DEFAULT_WEIGHTS = ROOT / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
DEFAULT_BOTTLES = ROOT / "data/processed/material_bottles/manifest.csv"


def fit_specialist(rows, val, weights, device, bottles):
    encoder = build_encoder()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    encoder.load_state_dict({k: v for k, v in state.items() if not k.startswith("fc.")})
    encoder = encoder.to(device).eval()

    def encode(samples):
        features = []
        with torch.inference_mode():
            for offset in range(0, len(samples), 32):
                tensors = []
                for row in samples[offset:offset + 32]:
                    with Image.open(ROOT / row["path"]) as image:
                        tensors.append(full_view(image.convert("RGB")))
                features.append(embeddings(encoder, torch.stack(tensors).to(device)).cpu().numpy())
        return np.concatenate(features)

    trash_train = [row for row in rows if row["split"] == "train" and row["label"] in {"glass", "plastic"}]
    bottle_train = [row for row in bottles if row["split"] == "train"]
    train = trash_train + bottle_train
    sample_weight = None
    if bottle_train:
        groups = [(items, mass) for items, mass in ((trash_train, .65), (bottle_train, .35))]
        sample_weight = []
        for items, mass in groups:
            count = {label: sum(row["label"] == label for row in items) for label in ("plastic", "glass")}
            if not min(count.values()):
                raise ValueError("Both materials must occur in each training source")
            sample_weight.extend(len(train) * mass / (2 * count[row["label"]]) for row in items)
    x = encode(train)
    scaler = StandardScaler().fit(x, sample_weight=sample_weight)
    head = LogisticRegression(C=0.03, max_iter=500).fit(scaler.transform(x), [row["label"] == "plastic" for row in train],
                                                       sample_weight=sample_weight)
    linear = head.coef_[0] / scaler.scale_
    bias = head.intercept_[0] - linear @ scaler.mean_
    val_odds = encode(val) @ linear + bias
    return encoder.cpu().state_dict(), linear, bias, val_odds, trash_train, bottle_train


def val_scores(rows, manifest_sha):
    scores = {}
    for key, checkpoint in CHECKPOINTS.items():
        root = RUNS / f"{key}_seed42_val"
        metrics = json.loads((root / "metrics.json").read_text())
        if (metrics["split"] != "val" or metrics["manifest_sha256"] != manifest_sha or
                metrics["checkpoint_sha256"] != file_sha256(checkpoint)):
            raise ValueError(f"Validation metrics and checkpoint differ for {key}")
        with (root / "predictions.csv").open(newline="") as stream:
            predictions = {row["path"]: row for row in csv.DictReader(stream)}
        if set(predictions) != {row["path"] for row in rows}:
            raise ValueError(f"Validation paths differ for {key}")
        aligned = [predictions[row["path"]] for row in rows]
        if any(record["split"] != "val" or record["true_label"] != row["label"]
               for record, row in zip(aligned, rows)):
            raise ValueError(f"Validation labels differ for {key}")
        scores[key] = np.asarray([[json.loads(row["scores_json"])[label] for label in LABELS] for row in aligned])
    return scores


def odds(scores):
    return np.log(scores[:, PLASTIC].clip(1e-20)) - np.log(scores[:, GLASS].clip(1e-20))


def variants(family):
    if family == "bottles":
        yield from variants("blend")
        yield from variants("agreement")
        return
    if family == "agreement":
        for peer, margin in product(PEERS, (0.0, 0.4, 0.8, 1.2)):
            yield {"specialist_weight": 0.50, "peer_id": peer, "peer_weight": 0.50,
                   "agreement_margin": margin}
        return
    for weight in (0.25, 0.50, 0.75, 1.0):
        yield {"specialist_weight": weight, "peer_id": None, "peer_weight": 0.0}
        for peer in PEERS:
            yield {"specialist_weight": 0.0, "peer_id": peer, "peer_weight": weight}
    for peer, (specialist_weight, peer_weight) in product(PEERS, ((.25, .25), (.50, .25), (.25, .50), (.50, .50))):
        yield {"specialist_weight": specialist_weight, "peer_id": peer, "peer_weight": peer_weight}


def infer_scores(checkpoint, rows, device):
    model, transform, _ = load_predictor(checkpoint, device)
    result = []
    with torch.inference_mode():
        for offset in range(0, len(rows), 24):
            tensors = []
            for row in rows[offset:offset + 24]:
                with Image.open(ROOT / row["path"]) as image:
                    tensors.append(transform(image.convert("RGB")))
            result.append(model(torch.stack(tensors).to(device)).softmax(dim=-1).cpu().numpy())
    return np.concatenate(result)


def load_bottles(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        if row["label"] not in {"glass", "plastic"} or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid bottle manifest row")
        if groups.setdefault(row["image_id"], row["split"]) != row["split"]:
            raise ValueError("The same TACO photo cannot occur in multiple splits")
    return rows


def combine(scores, specialist_odds, config):
    base = scores["e2"]
    strongest = base.argsort(axis=1)[:, -2:]
    mask = (strongest == GLASS).any(axis=1) & (strongest == PLASTIC).any(axis=1)
    specialist_weight, peer_weight = config["specialist_weight"], config["peer_weight"]
    logs = (1 - specialist_weight - peer_weight) * odds(base) + specialist_weight * specialist_odds
    if peer_weight:
        peer_odds = odds(scores[config["peer_id"]])
        logs += peer_weight * peer_odds
    margin = config.get("agreement_margin")
    if margin is not None:
        base_odds = odds(base)
        plastic = (base_odds <= 0) & (specialist_odds > margin) & (peer_odds > margin)
        glass = (base_odds > 0) & (specialist_odds < -margin) & (peer_odds < -margin)
        mask &= plastic | glass
    refined = base.copy()
    mass = base[mask, PLASTIC] + base[mask, GLASS]
    refined[mask, PLASTIC] = mass * expit(logs[mask])
    refined[mask, GLASS] = mass * expit(-logs[mask])
    return refined


def prediction_rows(val, scores):
    return [{"split": "val", "path": row["path"], "true_label": row["label"],
             "predicted_label": LABELS[score.argmax()], "confidence": float(score.max())}
            for row, score in zip(val, scores)]


def summarize(metrics):
    matrix = metrics["confusion_matrix"]
    return {"n": metrics["n"], "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "plastic_to_glass": matrix[PLASTIC][GLASS], "glass_to_plastic": matrix[GLASS][PLASTIC],
            "plastic_recall": metrics["classification_report"]["plastic"]["recall"],
            "glass_recall": metrics["classification_report"]["glass"]["recall"]}


def bottle_metrics(rows, scores, threshold=0.0):
    metrics = calculate_metrics(prediction_rows(rows, scores), threshold)
    summary = summarize(metrics)
    return {**summary, "balanced_accuracy": (summary["plastic_recall"] + summary["glass_recall"]) * .5,
            "support": {label: metrics["classification_report"][label]["support"] for label in ("plastic", "glass")},
            "coverage": metrics["coverage"], "accepted_accuracy": metrics["accepted_accuracy"]}


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def freeze_bottle_only(output, bottle_manifest, device):
    lock_path = output / "retrospective_lock.json"
    lock = json.loads(lock_path.read_text())
    parent = output / "best.pt"
    if lock["checkpoint_sha256"] != file_sha256(parent) or lock["taco_manifest_sha256"] != file_sha256(bottle_manifest):
        raise ValueError("The parent checkpoint and TACO manifest must match the frozen hashes")
    payload = torch.load(parent, map_location="cpu", weights_only=True)
    head = BottleModel()
    keys = head.state_dict()
    head.load_state_dict({key: payload["model_state"][key] for key in keys})
    checkpoint = output / "bottle_only.pt"
    bottle_payload = {"architecture": BOTTLE_ARCHITECTURE, "labels": list(LABELS), "smoke": False,
                      "seed": 42, "manifest_sha256": payload["manifest_sha256"],
                      "preprocessing": payload["preprocessing"], "model_state": head.state_dict()}
    val = [row for row in load_bottles(bottle_manifest) if row["split"] == "val"]
    head = head.to(device).eval()
    scores = []
    with torch.inference_mode():
        for row in val:
            with Image.open(ROOT / row["path"]) as image:
                tensor = full_view(image.convert("RGB")).unsqueeze(0).to(device)
            scores.append(head(tensor).softmax(dim=1).cpu().numpy()[0])
    sweep = sweep_validation(prediction_rows(val, np.asarray(scores)))
    bottle_payload["threshold"] = sweep["selected_threshold"]
    save_atomic(bottle_payload, checkpoint)
    proof = {"checkpoint_sha256": file_sha256(checkpoint), "taco_manifest_sha256": lock["taco_manifest_sha256"],
             "parent_checkpoint_sha256": lock["checkpoint_sha256"], "threshold": sweep["selected_threshold"],
             "taco_validation": bottle_metrics(val, np.asarray(scores), sweep["selected_threshold"]),
             "threshold_sweep": sweep}
    write_json(output / "bottle_only_lock.json", proof)
    print(json.dumps({k: v for k, v in proof.items() if k != "threshold_sweep"}, indent=2))


def fit(output, weights, device, family, bottle_manifest):
    output.mkdir(parents=True, exist_ok=False)
    manifest_sha = file_sha256(DEFAULT_MANIFEST)
    rows = read_manifest()
    val = [row for row in rows if row["split"] == "val"]
    scores = val_scores(val, manifest_sha)
    bottles = load_bottles(bottle_manifest) if family == "bottles" else []
    bottle_val = [row for row in bottles if row["split"] == "val"]
    if bottles and {row["label"] for row in bottle_val} != {"plastic", "glass"}:
        raise ValueError("Both materials must occur in TACO validation")
    encoder_state, linear, bias, all_odds, train, bottle_train = fit_specialist(rows, val + bottle_val, weights, device, bottles)
    specialist_odds, bottle_odds = all_odds[:len(val)], all_odds[len(val):]
    bottle_scores = {key: infer_scores(path, bottle_val, device) for key, path in CHECKPOINTS.items()} if bottles else {}
    bottle_base = bottle_metrics(bottle_val, bottle_scores["e2"]) if bottles else None
    base = summarize(calculate_metrics(prediction_rows(val, scores["e2"])))
    candidates = []
    for config in variants(family):
        if bottles and not config["specialist_weight"]:
            continue
        result = summarize(calculate_metrics(prediction_rows(val, combine(scores, specialist_odds, config))))
        if bottles:
            result["taco_validation"] = bottle_metrics(bottle_val, combine(bottle_scores, bottle_odds, config))
        candidates.append({**config, **result})

    def improves(row):
        if bottle_base is not None:
            return (row["plastic_to_glass"] < base["plastic_to_glass"]
                    and row["glass_to_plastic"] <= base["glass_to_plastic"] + 1
                    and row["macro_f1"] >= base["macro_f1"] - .003 and row["accuracy"] >= base["accuracy"] - 1 / base["n"]
                    and row["taco_validation"]["balanced_accuracy"] > bottle_base["balanced_accuracy"]
                    and row["taco_validation"]["plastic_recall"] >= bottle_base["plastic_recall"])
        return (row["plastic_to_glass"] < base["plastic_to_glass"]
                and row["glass_to_plastic"] <= base["glass_to_plastic"]
                and row["macro_f1"] >= base["macro_f1"] and row["accuracy"] >= base["accuracy"])

    def order(row):
        bottle_rank = -row["taco_validation"]["balanced_accuracy"] if bottles else 0
        return (bottle_rank, row["plastic_to_glass"] + row["glass_to_plastic"], row["plastic_to_glass"],
                -row["macro_f1"], -row.get("agreement_margin", 0.0),
                int(bool(row["specialist_weight"])) + int(bool(row["peer_weight"])))

    selected = min((row for row in candidates if improves(row)), key=order, default=None)
    report = {"family": family, "selection_split": "val", "train_split": "train", "specialist_train_n": len(train),
              "specialist_train_by_label": {label: sum(row["label"] == label for row in train)
                                            for label in ("plastic", "glass")},
              "imagenet_weights_sha256": file_sha256(weights), "manifest_sha256": manifest_sha,
              "selection_rule": "Reduce plastic->glass, do not increase glass->plastic, preserve six-class accuracy and macro-F1. Rank by total pair errors, plastic->glass, macro-F1, then auxiliary count.",
              "baseline": base, "selected": selected, "candidates": candidates}
    if bottles:
        report.update({"taco_baseline_validation": bottle_base, "taco_manifest_sha256": file_sha256(bottle_manifest),
                       "taco_train_by_label": {label: sum(row["label"] == label for row in bottle_train)
                                               for label in ("plastic", "glass")},
                       "selection_rule": "Improve TACO validation balanced accuracy and preserve plastic recall. On TrashNet val reduce plastic->glass, allow at most one additional reverse error and one fewer correct prediction, and macro-F1 loss at most .003. Rank by TACO balanced accuracy, then pair errors and macro-F1."})
    write_json(output / "validation_selection.json", report)
    if selected is None:
        print("No candidate met the validation criteria.")
        return
    config = {k: selected[k] for k in ("specialist_weight", "peer_id", "peer_weight", "agreement_margin") if k in selected}
    refined = prediction_rows(val, combine(scores, specialist_odds, config))
    threshold = sweep_validation(refined)
    write_json(output / "threshold_sweep.json", threshold)
    base_payload = torch.load(CHECKPOINTS["e2"], map_location="cpu", weights_only=True)
    model = MaterialModel(**config)
    model.base.load_state_dict(base_payload["model_state"])
    component_sha = {"e2": file_sha256(CHECKPOINTS["e2"])}
    if model.encoder is not None:
        model.encoder.load_state_dict(encoder_state)
        with torch.no_grad():
            model.head.weight.copy_(torch.as_tensor(linear, dtype=torch.float32).unsqueeze(0))
            model.head.bias.fill_(bias)
    if model.peer is not None:
        peer = config["peer_id"]
        peer_payload = torch.load(CHECKPOINTS[peer], map_location="cpu", weights_only=True)
        model.peer.load_state_dict(peer_payload["model_state"])
        component_sha[peer] = file_sha256(CHECKPOINTS[peer])
    best = output / "best.pt"
    save_atomic({"architecture": ARCHITECTURE, "labels": list(LABELS), "smoke": False, "seed": 42,
                 "manifest_sha256": manifest_sha, "preprocessing": base_payload["preprocessing"],
                 "material_config": config, "model_state": model.state_dict(), "component_sha256": component_sha,
                 "threshold": threshold["selected_threshold"]}, best)
    lock = {"allow_trashnet_test": True, "checkpoint_sha256": file_sha256(best), "manifest_sha256": manifest_sha,
            "threshold": threshold["selected_threshold"], "validation_selection_sha256": file_sha256(output / "validation_selection.json"),
            "note": "The older TrashNet test results were previously inspected. This is a retrospective check, kept separate from Phase 5."}
    if bottles:
        lock["taco_manifest_sha256"] = file_sha256(bottle_manifest)
    write_json(output / "retrospective_lock.json", lock)
    print(json.dumps({"baseline": base, "selected": selected, "threshold": threshold["selected_threshold"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "retrospective", "freeze-bottle-only", "bottles-test"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imagenet-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--family", choices=("blend", "agreement", "bottles"), default="blend")
    parser.add_argument("--bottle-manifest", type=Path, default=DEFAULT_BOTTLES)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    device = select_device(args.device)
    if args.mode == "fit":
        fit(args.output, args.imagenet_weights, device, args.family, args.bottle_manifest)
    elif args.mode == "freeze-bottle-only":
        freeze_bottle_only(args.output, args.bottle_manifest, device)
    else:
        lock_path = args.output / "retrospective_lock.json"
        lock = json.loads(lock_path.read_text())
        if lock["validation_selection_sha256"] != file_sha256(args.output / "validation_selection.json"):
            raise ValueError("Validation selection differs from the frozen record")
        if args.mode == "bottles-test":
            checkpoint = args.output / "best.pt"
            if (lock["taco_manifest_sha256"] != file_sha256(args.bottle_manifest) or
                    lock["checkpoint_sha256"] != file_sha256(checkpoint)):
                raise ValueError("TACO manifest and checkpoint must match their frozen hashes")
            rows = [row for row in load_bottles(args.bottle_manifest) if row["split"] == "test"]
            binary = args.output / "bottle_only.pt"
            binary_lock = json.loads((args.output / "bottle_only_lock.json").read_text())
            if (binary_lock["parent_checkpoint_sha256"] != lock["checkpoint_sha256"] or
                    binary_lock["taco_manifest_sha256"] != lock["taco_manifest_sha256"] or
                    binary_lock["checkpoint_sha256"] != file_sha256(binary)):
                raise ValueError("The binary comparison must be frozen before the TACO test")
            base = infer_scores(CHECKPOINTS["e2"], rows, device)
            refined = infer_scores(checkpoint, rows, device)
            bottle_only = infer_scores(binary, rows, device)
            restricted = np.zeros_like(base)
            restricted[:, (GLASS, PLASTIC)] = base[:, (GLASS, PLASTIC)] / base[:, (GLASS, PLASTIC)].sum(axis=1, keepdims=True)
            report = {"split": "test", "unit": "TACO ground-truth cropped bottle, split by source photo",
                      "source_photos": len({row["image_id"] for row in rows}),
                      "baseline": bottle_metrics(rows, base, .60), "refined": bottle_metrics(rows, refined, lock["threshold"]),
                      "baseline_restricted_to_pair": bottle_metrics(rows, restricted),
                      "bottle_only": bottle_metrics(rows, bottle_only, binary_lock["threshold"]),
                      "predictions": [{"path": row["path"], "label": row["label"], "baseline": LABELS[a.argmax()],
                                       "refined": LABELS[b.argmax()], "baseline_restricted": LABELS[c.argmax()],
                                       "bottle_only": LABELS[d.argmax()]}
                                      for row, a, b, c, d in zip(rows, base, refined, restricted, bottle_only)]}
            write_json(args.output / "taco_test.json", report)
            print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))
            return
        metrics = evaluate_split(args.output / "best.pt", "test", args.output / "retrospective_test",
                                 device=device, threshold=lock["threshold"], lock_path=lock_path)
        print(json.dumps(summarize(metrics), indent=2))


if __name__ == "__main__":
    main()
