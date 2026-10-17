from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from round2_common import load_strict, sha256, write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--qanet", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--b0-pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--artifact-count", type=int, default=10)
    return parser.parse_args()


class RunningStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.square = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, tensor):
        value = tensor.detach().float()
        self.count += value.numel()
        self.total += float(value.sum())
        self.square += float((value * value).sum())
        self.minimum = min(self.minimum, float(value.min()))
        self.maximum = max(self.maximum, float(value.max()))

    def row(self, scale, branch, name):
        mean = self.total / self.count
        variance = max(0.0, self.square / self.count - mean * mean)
        return {"scale": scale, "branch": branch, "tensor": name, "count": self.count,
                "min": self.minimum, "max": self.maximum, "mean": mean,
                "std": variance ** 0.5}


def gray_panel(path, title, size=(256, 192)):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("Cannot read {}".format(path))
    image = cv2.resize(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(image, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return image


def color_panel(path, title, size=(256, 192)):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Cannot read {}".format(path))
    image = cv2.resize(image, size)
    cv2.putText(image, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return image


def aux_arrays(aux):
    arrays = {}
    for scale, item in enumerate(aux["scales"]):
        for name in ("Avd", "Avt", "Adt", "Rv", "Rd", "Rt"):
            arrays["scale{}_{}".format(scale, name)] = item["ccc"][name].detach().cpu().numpy()[0, 0]
        for name in ("Wv", "Wd", "Wt"):
            arrays["scale{}_{}".format(scale, name)] = item["rtf"][name].detach().cpu().numpy()[0, 0]
    return arrays


def weight_means(aux):
    values = {}
    for scale, item in enumerate(aux["scales"]):
        for name in ("Wv", "Wd", "Wt"):
            values[(scale, name)] = float(item["rtf"][name].mean())
    return values


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    repo = args.repo.resolve()
    data_root = args.data_root.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "round2"))
    from lib.dataset import Data
    from QAnet import QAnet
    from round2.RACFNet import RACFNet

    qnet = QAnet().cuda()
    model = RACFNet("FULL").cuda()
    q_report = load_strict(qnet, args.qanet, "Round1 QAnet for formal inference")
    model_report = load_strict(model, args.model, "RACFNet FULL for formal inference")
    qnet.eval()
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.artifact_dir / "raw_tensors"
    map_dir = args.artifact_dir / "maps"
    visual_dir = args.artifact_dir / "visual_candidates"
    for folder in (raw_dir, map_dir, visual_dir):
        folder.mkdir(parents=True, exist_ok=True)

    test_root = data_root / "Test"
    loader = DataLoader(Data(str(test_root), mode="test"), batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    if len(loader) != 1000:
        raise RuntimeError("Expected 1000 test samples, got {}".format(len(loader)))
    accumulators = {}
    degradation_rows, inference_rows = [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for index, batch in enumerate(loader, 1):
            rgb, thermal, depth, _, _, (height, width), name = batch
            rgb = rgb.cuda(non_blocking=True).float()
            thermal = thermal.cuda(non_blocking=True).float()
            depth = depth.cuda(non_blocking=True).float()
            tic = time.perf_counter()
            dlogit, tlogit = qnet(rgb, thermal, depth)
            outputs, aux = model(rgb, thermal, depth, dlogit, tlogit, return_aux=True)
            for scale, item in enumerate(aux["scales"]):
                error = float((item["rtf"]["W"].sum(dim=1) - 1).abs().max())
                if error >= 1e-5:
                    raise RuntimeError("RTF Softmax error {} for {} scale {}".format(error, name[0], scale))
                for branch, names, source in (
                    ("CCC_CONSISTENCY", ("Avd", "Avt", "Adt"), item["ccc"]),
                    ("CCC_RELIABILITY", ("Rv", "Rd", "Rt"), item["ccc"]),
                    ("RTF_WEIGHT", ("Wv", "Wd", "Wt"), item["rtf"]),
                ):
                    for tensor_name in names:
                        key = (scale, branch, tensor_name)
                        accumulators.setdefault(key, RunningStats()).update(source[tensor_name])
            score = F.interpolate(outputs[-1], size=(int(height[0]), int(width[0])),
                                  mode="bilinear", align_corners=True)
            prediction = torch.sigmoid(score).squeeze().cpu().numpy()
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
            stem = Path(name[0]).stem
            prediction_path = args.output / (stem + ".png")
            cv2.imwrite(str(prediction_path), np.rint(prediction * 255).astype(np.uint8))
            torch.cuda.synchronize()
            inference_rows.append({"index": index, "image": stem,
                                   "seconds": time.perf_counter() - tic})

            if index <= args.artifact_count:
                arrays = aux_arrays(aux)
                np.savez_compressed(raw_dir / (stem + ".npz"), **arrays)
                for key, array in arrays.items():
                    cv2.imwrite(str(map_dir / (stem + "_" + key + ".png")),
                                np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))
                panels = [
                    color_panel(test_root / "V" / (stem + ".png"), "RGB"),
                    gray_panel(test_root / "D" / (stem + ".png"), "Depth"),
                    gray_panel(test_root / "T" / (stem + ".png"), "Thermal"),
                    gray_panel(test_root / "GT" / (stem + ".png"), "GT"),
                    gray_panel(args.b0_pred / (stem + ".png"), "QSFNet B0"),
                    gray_panel(prediction_path, "RACFNet FULL"),
                ]
                cv2.imwrite(str(visual_dir / (stem + "_comparison.png")),
                            np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:]))))

                base_means = weight_means(aux)
                variants = {
                    "rgb_zero": (torch.zeros_like(rgb), thermal, depth),
                    "depth_zero": (rgb, thermal, torch.zeros_like(depth)),
                    "thermal_zero": (rgb, torch.zeros_like(thermal), depth),
                }
                for degradation, (rv, tv, dv) in variants.items():
                    degraded_dlogit, degraded_tlogit = qnet(rv, tv, dv)
                    _, degraded_aux = model(rv, tv, dv, degraded_dlogit, degraded_tlogit,
                                            return_aux=True)
                    degraded_means = weight_means(degraded_aux)
                    for scale in range(3):
                        for weight_name in ("Wv", "Wd", "Wt"):
                            before = base_means[(scale, weight_name)]
                            after = degraded_means[(scale, weight_name)]
                            degradation_rows.append({
                                "image": stem, "degradation": degradation, "scale": scale,
                                "weight": weight_name, "before_mean": before,
                                "after_mean": after, "delta": after - before,
                            })
            print("{}/1000 {}".format(index, stem), flush=True)

    predictions = list(args.output.glob("*.png"))
    if len(predictions) != 1000:
        raise RuntimeError("Expected 1000 predictions, got {}".format(len(predictions)))
    reliability_rows = [stats.row(*key) for key, stats in sorted(accumulators.items())]
    write_csv(args.artifact_dir / "ROUND2_RELIABILITY_STATISTICS.csv", reliability_rows)
    write_csv(args.artifact_dir / "ROUND2_DEGRADATION_WEIGHT_CHANGES.csv", degradation_rows)
    write_csv(args.output / "inference_times.csv", inference_rows)
    manifest = {
        "status": "COMPLETE", "count": len(predictions),
        "seconds": time.perf_counter() - started,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "qanet_sha256": sha256(args.qanet), "model_sha256": sha256(args.model),
        "qanet_load": q_report, "model_load": model_report,
        "raw_tensor_sets": len(list(raw_dir.glob("*.npz"))),
        "visual_candidates": len(list(visual_dir.glob("*.png"))),
        "postprocess": "Round1 sigmoid + bilinear align_corners=True + per-image min-max + uint8 PNG",
    }
    write_json(args.output / "inference_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
