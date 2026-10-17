from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROUND2_SEED = 20261013


def seed_all(seed: int = ROUND2_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unwrap_state(obj):
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def load_strict(module, path: Path, label: str) -> dict:
    state = unwrap_state(torch.load(path, map_location="cpu"))
    result = module.load_state_dict(state, strict=True)
    return {
        "label": label,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "strict": True,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def load_round1_tnet(model, path: Path) -> dict:
    state = unwrap_state(torch.load(path, map_location="cpu"))
    result = model.load_state_dict(state, strict=False)
    allowed_prefixes = ("ccc_scales.", "rtf_scales.", "rtf_quality_v.")
    bad_missing = [name for name in result.missing_keys if not name.startswith(allowed_prefixes)]
    if result.unexpected_keys or bad_missing:
        raise RuntimeError(
            "Round1 Tnet transfer mismatch: unexpected={} bad_missing={}".format(
                list(result.unexpected_keys), bad_missing
            )
        )
    original_keys = set(state)
    model_keys = set(model.state_dict())
    matched = original_keys & model_keys
    return {
        "label": "Round1 Tnet transfer into RACFNet",
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "strict": False,
        "round1_key_count": len(original_keys),
        "matched_round1_keys": len(matched),
        "missing_new_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "all_round1_keys_reused": matched == original_keys,
    }


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, 1, 15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weight).sum(dim=(2, 3))
    union = ((prob + mask) * weight).sum(dim=(2, 3))
    return (wbce + 1 - (inter + 1) / (union - inter + 1)).mean()


def full_loss(outputs, edge, label):
    if len(outputs) != 14:
        raise RuntimeError("Expected 14 QSFNet outputs, got {}".format(len(outputs)))
    edge_loss = F.binary_cross_entropy_with_logits(outputs[0], edge, reduction="mean")
    unimodal = sum(structure_loss(x, label) for x in outputs[1:10])
    fused = 2 * structure_loss(outputs[13], label)
    fused = fused + sum(structure_loss(x, label) for x in outputs[10:13])
    return edge_loss + unimodal + fused


def tensor_stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach().float()
    return {
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "finite": bool(torch.isfinite(value).all()),
    }


def gradient_norm(named_parameters, prefix: str) -> tuple[float, int, int]:
    squared = 0.0
    tensors = 0
    nonzero = 0
    for name, parameter in named_parameters:
        if not name.startswith(prefix) or parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm())
        squared += norm * norm
        tensors += 1
        nonzero += int(norm > 0)
    return squared ** 0.5, tensors, nonzero


def parameter_snapshot(model, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    }


def parameter_change(before: dict[str, torch.Tensor], model) -> dict:
    after = dict(model.named_parameters())
    changed, max_delta, sum_delta = 0, 0.0, 0.0
    unchanged = []
    for name, old in before.items():
        delta = (after[name].detach().cpu() - old).abs()
        local_max = float(delta.max())
        changed += int(local_max > 0)
        if local_max == 0:
            unchanged.append(name)
        max_delta = max(max_delta, local_max)
        sum_delta += float(delta.sum())
    return {
        "tracked_parameter_tensors": len(before),
        "changed_parameter_tensors": changed,
        "max_abs_delta": max_delta,
        "sum_abs_delta": sum_delta,
        "unchanged_parameter_names": unchanged,
    }


def bn_snapshot(module) -> dict:
    result = {}
    for name, layer in module.named_modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            result[name] = {
                "mean": layer.running_mean.detach().cpu().clone(),
                "var": layer.running_var.detach().cpu().clone(),
                "n": int(layer.num_batches_tracked),
            }
    return result


def bn_diff(before: dict, after: dict) -> dict:
    changed, max_mean, max_var, max_n = 0, 0.0, 0.0, 0
    for name, old in before.items():
        new = after[name]
        mean_delta = float((new["mean"] - old["mean"]).abs().max())
        var_delta = float((new["var"] - old["var"]).abs().max())
        n_delta = int(new["n"] - old["n"])
        changed += int(mean_delta > 0 or var_delta > 0 or n_delta > 0)
        max_mean = max(max_mean, mean_delta)
        max_var = max(max_var, var_delta)
        max_n = max(max_n, n_delta)
    return {
        "bn_layers": len(before),
        "changed_layers": changed,
        "max_running_mean_delta": max_mean,
        "max_running_var_delta": max_var,
        "max_num_batches_tracked_delta": max_n,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def atomic_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def validate_train_data(data_root: Path) -> Path:
    train = data_root.resolve() / "Train"
    for name in ("V", "D", "T", "GT", "eg"):
        folder = train / name
        count = len(list(folder.glob("*.png")))
        if count != 1048:
            raise RuntimeError("Expected 1048 PNG files in {}, got {}".format(folder, count))
    return train
