from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from round4_common import (
    DEGRADATION_LEVELS, HISTORICAL, LEVEL_NAMES, METRICS, MODEL_ORDER,
    DegradationDataset, OnlineMoments, configure_imports, disk_gib, load_model,
    metric_measure, model_specs, per_image_metric_with_curves, predict_batch,
    seed_all, strict_load, write_csv, write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="Project root containing checkpoints and data references")
    p.add_argument("--repo", type=Path, required=True,
                   help="RACFNet source repository root")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for Round4 robustness results")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", choices=MODEL_ORDER)
    return p.parse_args()


def collate(items):
    return (torch.stack([x[0] for x in items]), torch.stack([x[1] for x in items]),
            torch.stack([x[2] for x in items]), [x[3] for x in items],
            [x[4] for x in items], [x[5] for x in items], [x[6] for x in items])


class MeanAccumulator:
    def __init__(self):
        self.values = []

    def add(self, tensor):
        if torch.is_tensor(tensor):
            x = tensor.detach().float().reshape(-1).cpu().numpy()
            self.values.extend(float(v) for v in x)
        else:
            self.values.append(float(tensor))

    def result(self):
        x = np.asarray(self.values, np.float64)
        return {"sample_count": len(x), "mean": float(x.mean()), "std": float(x.std()),
                "min": float(x.min()), "max": float(x.max())}


def mechanism_collect(aux, ccc_capture, stores):
    ccc_tensor, ccc_spatial = stores["ccc_tensor"], stores["ccc_spatial"]
    rtf_tensor, rtf_spatial = stores["rtf_tensor"], stores["rtf_spatial"]
    entropy_store, dominant_store = stores["entropy"], stores["dominant"]
    sat_low, sat_high = stores["sat_low"], stores["sat_high"]
    residual = stores["residual"]
    for scale, item in enumerate(aux["scales"]):
        if item["ccc"] is not None:
            for tensor_name in ("Rv", "Rd", "Rt"):
                tensor = item["ccc"][tensor_name]
                ccc_tensor[(scale, tensor_name)].add_tensor(tensor)
                ccc_spatial[(scale, tensor_name)].add(tensor.float().var(dim=(2, 3), unbiased=False))
            captured = ccc_capture[scale]
            if captured is not None:
                for branch, raw, cal in zip(("v", "d", "t"), captured["raw"], captured["cal"]):
                    delta = cal - raw
                    dn = delta.float().flatten(1).norm(dim=1)
                    bn = raw.float().flatten(1).norm(dim=1).clamp_min(1e-12)
                    residual[(scale, "CCC", branch)].add(dn / bn)
        if item["rtf"] is not None:
            rtf = item["rtf"]
            weights = rtf["W"].float()
            dominant = weights.argmax(dim=1)
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
            entropy_store[scale].add(entropy.mean(dim=(1, 2)))
            for ci, tensor_name in enumerate(("Wv", "Wd", "Wt")):
                tensor = rtf[tensor_name]
                rtf_tensor[(scale, tensor_name)].add_tensor(tensor)
                rtf_spatial[(scale, tensor_name)].add(tensor.float().var(dim=(2, 3), unbiased=False))
                dominant_store[(scale, tensor_name)].add((dominant == ci).float().mean(dim=(1, 2)))
                sat_low[(scale, tensor_name)].add((tensor < 1e-4).float().mean(dim=(1, 2, 3)))
                sat_high[(scale, tensor_name)].add((tensor > .9999).float().mean(dim=(1, 2, 3)))
            for branch, base, new in (("d", item["X_d_base"], item["X_d_new"]),
                                      ("t", item["X_t_base"], item["X_t_new"])):
                dn = (new - base).float().flatten(1).norm(dim=1)
                bn = base.float().flatten(1).norm(dim=1).clamp_min(1e-12)
                residual[(scale, "RTF", branch)].add(dn / bn)


def mechanism_rows(model_name, condition, level, stores):
    ccc, weights, entropy, residuals = [], [], [], []
    for (scale, tensor), stat in stores["ccc_tensor"].items():
        row = {"model": model_name, "condition": condition, "level": level,
               "scale": scale, "tensor": tensor, **stat.result()}
        row.update({f"spatial_{k}": v for k, v in stores["ccc_spatial"][(scale, tensor)].result().items()})
        ccc.append(row)
    for (scale, tensor), stat in stores["rtf_tensor"].items():
        row = {"model": model_name, "condition": condition, "level": level,
               "scale": scale, "tensor": tensor, **stat.result()}
        row.update({f"spatial_{k}": v for k, v in stores["rtf_spatial"][(scale, tensor)].result().items()})
        row["dominant_area_mean"] = stores["dominant"][(scale, tensor)].result()["mean"]
        row["saturated_low_mean"] = stores["sat_low"][(scale, tensor)].result()["mean"]
        row["saturated_high_mean"] = stores["sat_high"][(scale, tensor)].result()["mean"]
        weights.append(row)
    for scale, stat in stores["entropy"].items():
        entropy.append({"model": model_name, "condition": condition, "level": level,
                        "scale": scale, **stat.result(), "maximum_entropy": float(np.log(3))})
    for (scale, module, branch), stat in stores["residual"].items():
        residuals.append({"model": model_name, "condition": condition, "level": level,
                          "scale": scale, "module": module, "branch": branch,
                          "measure": "delta_norm_over_base_norm", **stat.result()})
    return ccc, weights, entropy, residuals


def condition_paths(cache: Path, model: str, condition: str, level: str):
    slug = f"{model.lower()}__{condition}__{level}"
    return (cache / "summaries" / f"{slug}.json", cache / "per_image" / f"{slug}.csv",
            cache / "mechanism" / f"{slug}.json")


def run_condition(model_name, condition, level, model, qnet, test_root, cache,
                  batch_size, workers):
    summary_path, per_path, mechanism_path = condition_paths(cache, model_name, condition, level)
    if summary_path.exists() and per_path.exists() and mechanism_path.exists():
        return "SKIPPED_COMPLETE"
    for p in (summary_path, per_path, mechanism_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    dataset = DegradationDataset(test_root, condition, level)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0, collate_fn=collate)
    ccc_capture = [None, None, None]
    hooks = []
    if model_name in ("CCC_ONLY", "FULL"):
        for idx, module in enumerate(model.ccc_scales):
            def hook(_module, inputs, output, index=idx):
                ccc_capture[index] = {"raw": inputs[:3], "cal": output[:3]}
            hooks.append(module.register_forward_hook(hook))
    stores = {
        "ccc_tensor": defaultdict(OnlineMoments), "ccc_spatial": defaultdict(MeanAccumulator),
        "rtf_tensor": defaultdict(OnlineMoments), "rtf_spatial": defaultdict(MeanAccumulator),
        "entropy": defaultdict(MeanAccumulator), "dominant": defaultdict(MeanAccumulator),
        "sat_low": defaultdict(MeanAccumulator), "sat_high": defaultdict(MeanAccumulator),
        "residual": defaultdict(MeanAccumulator),
    }
    per_rows = []
    scalar_sum = np.zeros(8, np.float64)
    fcurve_sum = np.zeros(256, np.float64)
    ecurve_sum = np.zeros(256, np.float64)
    count = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, (rgb, thermal, depth, gts, hs, ws, stems) in enumerate(loader, 1):
            rgb, thermal, depth = (x.cuda(non_blocking=True) for x in (rgb, thermal, depth))
            preds, aux = predict_batch(model, qnet, rgb, thermal, depth,
                                       list(zip(hs, ws)), model_name == "B0",
                                       return_aux=model_name != "B0")
            if aux is not None:
                mechanism_collect(aux, ccc_capture, stores)
            for pred, gt, stem in zip(preds, gts, stems):
                metrics, fcurve, ecurve = per_image_metric_with_curves(pred, gt)
                scalar_sum += np.asarray([metrics[x] for x in METRICS], np.float64)
                fcurve_sum += fcurve
                ecurve_sum += ecurve
                count += 1
                per_rows.append({"model": model_name, "condition": condition, "level": level,
                                 "sample_id": stem, **metrics})
            if batch_index % 50 == 0:
                print(json.dumps({"model": model_name, "condition": condition, "level": level,
                                  "processed": count, "total": len(dataset)}), flush=True)
    torch.cuda.synchronize()
    for handle in hooks:
        handle.remove()
    if count != 1000:
        raise RuntimeError(f"condition incomplete: {model_name} {condition} {level} {count}")
    mean_scalar = scalar_sum / count
    fcurve = fcurve_sum / count
    ecurve = ecurve_sum / count
    aggregate = dict(zip(METRICS, mean_scalar.tolist()))
    aggregate.update({"E_mean": float(ecurve.mean()), "E_max": float(ecurve.max()),
                      "F_mean": float(fcurve.mean()), "F_max": float(fcurve.max())})
    summary = {"model": model_name, "condition": condition, "level": level,
               "sample_count": count, **aggregate,
               "wall_seconds": time.perf_counter() - started,
               "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
               "finished_utc": datetime.now(timezone.utc).isoformat()}
    ccc, weights, entropy, residuals = mechanism_rows(model_name, condition, level, stores)
    write_csv(per_path, per_rows)
    write_json(summary_path, summary)
    write_json(mechanism_path, {"ccc": ccc, "weights": weights,
                                "entropy": entropy, "residuals": residuals})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return "COMPLETE"


def load_all_rows(cache: Path):
    summaries, per_rows, ccc, weights, entropy, residuals = [], [], [], [], [], []
    for path in sorted((cache / "summaries").glob("*.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    for path in sorted((cache / "per_image").glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            per_rows.extend(csv.DictReader(stream))
    for path in sorted((cache / "mechanism").glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        ccc.extend(obj["ccc"]); weights.extend(obj["weights"])
        entropy.extend(obj["entropy"]); residuals.extend(obj["residuals"])
    return summaries, per_rows, ccc, weights, entropy, residuals


def finalize(out: Path, cache: Path):
    summaries, per_rows, ccc, weights, entropy, residuals = load_all_rows(cache)
    # B0 clean does not require a redundant forward pass.
    summaries.append({"model": "B0", "condition": "clean", "level": "clean",
                      "sample_count": 1000, **dict(zip(METRICS, HISTORICAL["B0"])),
                      "source": "Round1 formal predictions"})
    write_csv(out / "ROUND4_DEGRADATION_SUMMARY.csv", summaries)
    write_csv(out / "ROUND4_DEGRADATION_PER_IMAGE.csv", per_rows)
    write_csv(out / "ROUND4_CCC_RELIABILITY.csv", ccc)
    write_csv(out / "ROUND4_RTF_WEIGHTS.csv", weights)
    write_csv(out / "ROUND4_WEIGHT_ENTROPY.csv", entropy)
    write_csv(out / "ROUND4_RESIDUAL_ANALYSIS.csv", residuals)
    lookup = {(x["model"], x["condition"], x["level"]): x for x in summaries}
    deltas = []
    for row in summaries:
        if row["condition"] == "clean":
            continue
        clean = lookup[(row["model"], "clean", "clean")]
        delta = {"model": row["model"], "condition": row["condition"], "level": row["level"]}
        for metric in METRICS:
            delta[f"{metric}_degraded_minus_clean"] = float(row[metric]) - float(clean[metric])
        for other in MODEL_ORDER:
            if other == row["model"]:
                continue
            key = (other, row["condition"], row["level"])
            if key in lookup:
                for metric in METRICS:
                    delta[f"{metric}_{row['model']}_minus_{other}"] = float(row[metric]) - float(lookup[key][metric])
        deltas.append(delta)
    write_csv(out / "ROUND4_DEGRADATION_DELTA.csv", deltas)
    clean_weights = {(x["model"], int(x["scale"]), x["tensor"]): x for x in weights
                     if x["condition"] == "clean"}
    perf_delta = {(x["model"], x["condition"], x["level"]): x for x in deltas}
    shifts = []
    for row in weights:
        if row["condition"] == "clean":
            continue
        key = (row["model"], int(row["scale"]), row["tensor"])
        if key not in clean_weights:
            continue
        base = clean_weights[key]
        shift = {"model": row["model"], "condition": row["condition"], "level": row["level"],
                 "scale": row["scale"], "tensor": row["tensor"],
                 "degraded_mean": row["mean"], "clean_mean": base["mean"],
                 "weight_mean_shift": float(row["mean"]) - float(base["mean"]),
                 "degraded_spatial_variance": row.get("spatial_mean"),
                 "clean_spatial_variance": base.get("spatial_mean"),
                 "dominant_area_shift": float(row["dominant_area_mean"]) - float(base["dominant_area_mean"]),
                 "saturated_high_shift": float(row["saturated_high_mean"]) - float(base["saturated_high_mean"])}
        p = perf_delta.get((row["model"], row["condition"], row["level"]), {})
        for metric in METRICS:
            shift[f"{metric}_degraded_minus_clean"] = p.get(f"{metric}_degraded_minus_clean")
        shifts.append(shift)
    write_csv(out / "ROUND4_DEGRADATION_WEIGHT_SHIFT.csv", shifts)


def main():
    a = parse_args()
    root, repo, out = a.root.resolve(), a.repo.resolve(), a.out.resolve()
    state_path = out / "pipeline_state.json"
    estimate = json.loads((out / "ROUND4_COST_ESTIMATE.json").read_text(encoding="utf-8"))
    if estimate["status"] != "SAFE_TO_CONTINUE":
        raise RuntimeError("preflight did not authorize full degradation evaluation")
    if disk_gib(root)["free_gib"] < 10 or disk_gib(Path("/"))["free_gib"] < 10:
        raise RuntimeError("disk safety guard failed before degradation")
    seed_all()
    QAnet, Tnet, RACFNet = configure_imports(repo)
    specs = model_specs(root)
    qnet = QAnet().cuda().eval()
    strict_load(qnet, root / "checkpoints/QAnet_final.pth")
    test_root = root / "datasets/VDT-2048/Test"
    cache = out / "cache/degradation"
    models = (a.model,) if a.model else MODEL_ORDER
    for model_name in models:
        model = load_model(model_name, specs, Tnet, RACFNet)
        conditions = [(name, level) for name in DEGRADATION_LEVELS for level in LEVEL_NAMES]
        if model_name != "B0":
            conditions.insert(0, ("clean", "clean"))
        for condition, level in conditions:
            write_json(state_path, {"phase": "PHASE_4_DEGRADATION", "status": "RUNNING",
                                    "model": model_name, "condition": condition, "level": level,
                                    "timestamp_utc": datetime.now(timezone.utc).isoformat()})
            run_condition(model_name, condition, level, model, qnet, test_root, cache,
                          a.batch_size, a.workers)
            if disk_gib(root)["free_gib"] < 10:
                raise RuntimeError("data disk guard failed during degradation")
        del model
        torch.cuda.empty_cache()
    finalize(out, cache)
    state = {"phase": "PHASE_5_MECHANISM_COMPLETE", "status": "PASS",
             "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
