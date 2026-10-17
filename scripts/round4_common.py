from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from py_sod_metrics import Emeasure, Fmeasure, MAE, Smeasure
from torch.utils.data import Dataset


SEED = 20261013
METRICS = ("S_alpha", "MAE", "E_adp", "E_mean", "E_max", "F_adp", "F_mean", "F_max")
MODEL_ORDER = ("B0", "CCC_ONLY", "RTF_ONLY", "FULL")
ATTRIBUTES = {
    "Visible": ("V-BSO", "V-LI", "V-MSO", "V-NI", "V-SA", "V-SI", "V-SSO"),
    "Depth": ("D-BI", "D-BM", "D-II", "D-SSO"),
    "Thermal": ("T-Cr", "T-HR", "T-RD"),
}
HISTORICAL = {
    "B0": (0.9396123173454957, 0.0021581621859681375, 0.9869069477789053,
           0.98830455257906, 0.9919698196572646, 0.8902727104010656,
           0.9058131423657649, 0.9218892907870373),
    "CCC_ONLY": (0.9398584896130541, 0.002143811504289216,
                 0.988546671509267, 0.9885481613276048, 0.9920697831270924,
                 0.8947484319849044, 0.9086292066714317, 0.9226145791380878),
    "RTF_ONLY": (0.9405550378859538, 0.0021284086499183006,
                 0.9884236700353429, 0.988983262396542, 0.9925291085827986,
                 0.8935744538292675, 0.9082043024133586, 0.9235691085739439),
    "FULL": (0.9406192455793603, 0.0021345483430351307,
             0.9889208441854751, 0.988745994252205, 0.9922832538611635,
             0.8960823477949789, 0.9101882051847916, 0.9237554238188649),
}


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def disk_gib(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {"total_gib": usage.total / 2**30, "used_gib": usage.used / 2**30,
            "free_gib": usage.free / 2**30}


def unwrap_state(obj):
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def strict_load(module: torch.nn.Module, path: Path) -> dict:
    state = unwrap_state(torch.load(path, map_location="cpu"))
    result = module.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict load mismatch for {path}")
    return {"strict": True, "missing_keys": [], "unexpected_keys": []}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    round: int
    checkpoint: Path
    epoch: int
    step: int
    ccc: bool
    rtf: bool
    prediction_dir: Path


def model_specs(root: Path) -> dict[str, ModelSpec]:
    return {
        "B0": ModelSpec("B0", 1, root / "checkpoints/Tnet_final.pth", 200, 52400,
                        False, False, root / "predictions/round1_reproduction"),
        "CCC_ONLY": ModelSpec("CCC_ONLY", 3,
                              root / "round3/checkpoints/ccc_only/RACFNet_CCC_ONLY_final.pth",
                              200, 52400, True, False,
                              root / "round3/predictions/ccc_only"),
        "RTF_ONLY": ModelSpec("RTF_ONLY", 3,
                              root / "round3/checkpoints/rtf_only/RACFNet_RTF_ONLY_final.pth",
                              200, 52400, False, True,
                              root / "round3/predictions/rtf_only"),
        "FULL": ModelSpec("FULL", 2,
                          root / "round2/checkpoints/RACFNet_FULL_final.pth",
                          200, 52400, True, True,
                          root / "round2/predictions/full"),
    }


def configure_imports(repo: Path):
    sys.path[:0] = [str(repo), str(repo / "round2")]
    from QAnet import QAnet
    from Tnet import Tnet
    from round2.RACFNet import RACFNet
    return QAnet, Tnet, RACFNet


def make_model(name: str, Tnet, RACFNet):
    return Tnet() if name == "B0" else RACFNet(name)


def load_model(name: str, specs: dict[str, ModelSpec], Tnet, RACFNet,
               device: str = "cuda") -> torch.nn.Module:
    model = make_model(name, Tnet, RACFNet)
    strict_load(model, specs[name].checkpoint)
    return model.to(device).eval()


def metric_measure(pairs: Iterable[tuple[np.ndarray, np.ndarray]],
                   return_curves: bool = False):
    fm, em, sm, mae = Fmeasure(), Emeasure(), Smeasure(), MAE()
    count = 0
    for pred, gt in pairs:
        fm.step(pred=pred, gt=gt)
        em.step(pred=pred, gt=gt)
        sm.step(pred=pred, gt=gt)
        mae.step(pred=pred, gt=gt)
        count += 1
    if not count:
        raise RuntimeError("empty metric set")
    f = fm.get_results()["fm"]
    e = em.get_results()["em"]
    result = {
        "S_alpha": float(sm.get_results()["sm"]),
        "MAE": float(mae.get_results()["mae"]),
        "E_adp": float(e["adp"]),
        "E_mean": float(np.mean(e["curve"])),
        "E_max": float(np.max(e["curve"])),
        "F_adp": float(f["adp"]),
        "F_mean": float(np.mean(f["curve"])),
        "F_max": float(np.max(f["curve"])),
    }
    if return_curves:
        return result, np.asarray(f["curve"], np.float32), np.asarray(e["curve"], np.float32)
    return result


def read_pair(pred_path: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if pred is None or gt is None or pred.shape != gt.shape:
        raise RuntimeError(f"invalid prediction/GT pair: {pred_path}, {gt_path}")
    return pred, gt


def validate_prediction_dir(path: Path, gt_dir: Path) -> list[str]:
    preds = {x.stem: x for x in path.glob("*.png")}
    gts = {x.stem: x for x in gt_dir.glob("*.png")}
    if len(preds) != 1000 or set(preds) != set(gts):
        raise RuntimeError(f"prediction mismatch at {path}: pred={len(preds)} gt={len(gts)}")
    for stem in sorted(preds):
        pred, gt = read_pair(preds[stem], gts[stem])
        if pred.min() == pred.max():
            raise RuntimeError(f"constant prediction {path}/{stem}.png")
    return sorted(gts)


def evaluate_prediction_dir(path: Path, gt_dir: Path, stems: Iterable[str] | None = None):
    ids = list(stems) if stems is not None else validate_prediction_dir(path, gt_dir)
    return metric_measure(read_pair(path / f"{stem}.png", gt_dir / f"{stem}.png") for stem in ids)


def per_image_metric_with_curves(pred: np.ndarray, gt: np.ndarray):
    return metric_measure([(pred, gt)], return_curves=True)


MEAN_RGB = np.array([[[0.391 * 255, 0.363 * 255, 0.338 * 255]]], np.float32)
MEAN_T = np.array([[[0.170 * 255, 0.403 * 255, 0.556 * 255]]], np.float32)
MEAN_D = np.array([[[0.034 * 255, 0.034 * 255, 0.034 * 255]]], np.float32)
STD_RGB = np.array([[[0.224 * 255, 0.217 * 255, 0.206 * 255]]], np.float32)
STD_T = np.array([[[0.160 * 255, 0.188 * 255, 0.238 * 255]]], np.float32)
STD_D = np.array([[[0.007 * 255, 0.007 * 255, 0.007 * 255]]], np.float32)


DEGRADATION_LEVELS = {
    "rgb_brightness": ({"factor": 0.75}, {"factor": 0.50}, {"factor": 0.25}),
    "rgb_noise": ({"sigma": 5.0}, {"sigma": 15.0}, {"sigma": 30.0}),
    "rgb_blur": ({"kernel": 5, "sigma": 1.0}, {"kernel": 11, "sigma": 3.0},
                 {"kernel": 21, "sigma": 5.0}),
    "depth_holes": ({"probability": 0.05}, {"probability": 0.15}, {"probability": 0.30}),
    "depth_block": ({"area_ratio": 0.05}, {"area_ratio": 0.15}, {"area_ratio": 0.30}),
    "depth_noise": ({"sigma": 2.0}, {"sigma": 4.0}, {"sigma": 8.0}),
    "thermal_blur": ({"kernel": 5, "sigma": 1.0}, {"kernel": 11, "sigma": 3.0},
                     {"kernel": 21, "sigma": 5.0}),
    "thermal_noise": ({"sigma": 5.0}, {"sigma": 15.0}, {"sigma": 30.0}),
    "thermal_local_heat": ({"radius_fraction": 0.08, "amplitude": 20.0},
                           {"radius_fraction": 0.15, "amplitude": 50.0},
                           {"radius_fraction": 0.25, "amplitude": 90.0}),
}
LEVEL_NAMES = ("light", "medium", "heavy")


def _rng_for(stem: str, condition: str, level: str) -> np.random.Generator:
    token = f"{SEED}|{stem}|{condition}|{level}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(value)


def _noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape), 0, 255)


def apply_degradation(rgb: np.ndarray, thermal: np.ndarray, depth: np.ndarray,
                      stem: str, condition: str, level: str):
    if condition == "clean":
        return rgb, thermal, depth
    idx = LEVEL_NAMES.index(level)
    p = DEGRADATION_LEVELS[condition][idx]
    rng = _rng_for(stem, condition, level)
    rgb, thermal, depth = rgb.copy(), thermal.copy(), depth.copy()
    if condition == "rgb_brightness":
        rgb = np.clip(rgb.astype(np.float32) * p["factor"], 0, 255)
    elif condition == "rgb_noise":
        rgb = _noise(rgb, p["sigma"], rng)
    elif condition == "rgb_blur":
        rgb = cv2.GaussianBlur(rgb, (p["kernel"], p["kernel"]), p["sigma"])
    elif condition == "depth_holes":
        mask = rng.random(depth.shape[:2]) < p["probability"]
        depth[mask] = 0
    elif condition == "depth_block":
        h, w = depth.shape[:2]
        bh = max(1, int(round(h * math.sqrt(p["area_ratio"]))))
        bw = max(1, int(round(w * math.sqrt(p["area_ratio"]))))
        y = int(rng.integers(0, max(1, h - bh + 1)))
        x = int(rng.integers(0, max(1, w - bw + 1)))
        depth[y:y + bh, x:x + bw] = 0
    elif condition == "depth_noise":
        depth = _noise(depth, p["sigma"], rng)
    elif condition == "thermal_blur":
        thermal = cv2.GaussianBlur(thermal, (p["kernel"], p["kernel"]), p["sigma"])
    elif condition == "thermal_noise":
        thermal = _noise(thermal, p["sigma"], rng)
    elif condition == "thermal_local_heat":
        h, w = thermal.shape[:2]
        cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
        radius = max(2, int(min(h, w) * p["radius_fraction"]))
        yy, xx = np.ogrid[:h, :w]
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (radius / 2) ** 2))
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        thermal = np.clip(thermal.astype(np.float32) + sign * p["amplitude"] * blob[..., None], 0, 255)
    else:
        raise ValueError(condition)
    return rgb.astype(np.float32), thermal.astype(np.float32), depth.astype(np.float32)


def normalize_resize(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    x = (image.astype(np.float32) - mean) / std
    x = cv2.resize(x, (384, 384), interpolation=cv2.INTER_LINEAR)
    if x.ndim == 2:
        x = x[..., None]
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).float()


class DegradationDataset(Dataset):
    def __init__(self, test_root: Path, condition: str, level: str):
        self.root = test_root
        self.condition = condition
        self.level = level
        self.stems = sorted(p.stem for p in (test_root / "GT").glob("*.png"))

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index):
        stem = self.stems[index]
        rgb = cv2.imread(str(self.root / "V" / f"{stem}.png"), cv2.IMREAD_COLOR).astype(np.float32)
        thermal = cv2.imread(str(self.root / "T" / f"{stem}.png"), cv2.IMREAD_COLOR).astype(np.float32)
        depth = cv2.imread(str(self.root / "D" / f"{stem}.png"), cv2.IMREAD_COLOR).astype(np.float32)
        gt = cv2.imread(str(self.root / "GT" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if any(x is None for x in (rgb, thermal, depth, gt)):
            raise RuntimeError(f"unreadable sample {stem}")
        h, w = gt.shape
        rgb, thermal, depth = apply_degradation(rgb, thermal, depth, stem, self.condition, self.level)
        return (normalize_resize(rgb, MEAN_RGB, STD_RGB),
                normalize_resize(thermal, MEAN_T, STD_T),
                normalize_resize(depth, MEAN_D, STD_D), gt, h, w, stem)


def predict_batch(model, qnet, rgb, thermal, depth, sizes, is_b0: bool,
                  return_aux: bool = False):
    dlogit, tlogit = qnet(rgb, thermal, depth)
    if return_aux and not is_b0:
        outputs, aux = model(rgb, thermal, depth, dlogit, tlogit, return_aux=True)
    else:
        outputs = model(rgb, thermal, depth, dlogit, tlogit)
        aux = None
    probs = []
    for i, (h, w) in enumerate(sizes):
        score = F.interpolate(outputs[-1][i:i + 1], size=(int(h), int(w)),
                              mode="bilinear", align_corners=True)
        pred = torch.sigmoid(score).squeeze().detach().cpu().numpy()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        probs.append(np.rint(pred * 255).astype(np.uint8))
    return probs, aux


class OnlineMoments:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def add_tensor(self, tensor: torch.Tensor) -> None:
        x = tensor.detach().float()
        self.count += x.numel()
        self.sum += float(x.sum())
        self.sum_sq += float((x * x).sum())
        self.minimum = min(self.minimum, float(x.min()))
        self.maximum = max(self.maximum, float(x.max()))

    def result(self) -> dict:
        mean = self.sum / max(self.count, 1)
        variance = max(0.0, self.sum_sq / max(self.count, 1) - mean * mean)
        return {"count": self.count, "mean": mean, "std": math.sqrt(variance),
                "min": self.minimum, "max": self.maximum, "variance": variance}

