from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("CCC_ONLY", "RTF_ONLY"))
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--qanet", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--b0-pred", type=Path, required=True)
    p.add_argument("--full-pred", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--artifact-count", type=int, default=10)
    return p.parse_args()


def stats(t):
    x = t.detach().float()
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
            "std": float(x.std(unbiased=False)), "finite": bool(torch.isfinite(x).all())}


def norm_ratio(delta, base):
    dn = float(delta.detach().float().norm()); bn = float(base.detach().float().norm())
    return dn, bn, dn / (bn + 1e-12)


def panel(path, title, color=False, size=(256, 192)):
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), flag)
    if image is None: raise RuntimeError("Cannot read {}".format(path))
    image = cv2.resize(image, size)
    if not color: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(image, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
    return image


def predict(model, qnet, rgb, thermal, depth, height=None, width=None):
    dl, tl = qnet(rgb, thermal, depth)
    outputs, aux = model(rgb, thermal, depth, dl, tl, return_aux=True)
    score = outputs[-1]
    if height is not None:
        score = F.interpolate(score, size=(int(height), int(width)), mode="bilinear", align_corners=True)
    prob = torch.sigmoid(score)
    return prob, aux


def main():
    a = parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
    repo = a.repo.resolve(); sys.path[:0] = [str(repo), str(repo / "round2")]
    from lib.dataset import Data
    from QAnet import QAnet
    from round2.RACFNet import RACFNet
    from round2.round2_common import load_strict, sha256, write_csv, write_json
    model, qnet = RACFNet(a.mode).cuda(), QAnet().cuda()
    q_report = load_strict(qnet, a.qanet, "Round1 QAnet formal inference")
    m_report = load_strict(model, a.model, a.mode + " formal inference")
    qnet.eval(); model.eval(); a.output.mkdir(parents=True, exist_ok=True)
    raw_dir, map_dir, visual_dir = (a.artifact_dir / x for x in ("raw_tensors", "maps", "visual_candidates"))
    for d in (raw_dir, map_dir, visual_dir): d.mkdir(parents=True, exist_ok=True)

    ccc_capture = [None, None, None]
    handles = []
    if a.mode == "CCC_ONLY":
        for idx, module in enumerate(model.ccc_scales):
            def hook(_module, inputs, output, index=idx):
                ccc_capture[index] = {"raw": inputs[:3], "cal": output[:3]}
            handles.append(module.register_forward_hook(hook))

    test_root = a.data_root.resolve() / "Test"
    loader = DataLoader(Data(str(test_root), mode="test"), batch_size=1, shuffle=False,
                        num_workers=a.workers, pin_memory=True)
    if len(loader) != 1000: raise RuntimeError("Expected 1000 samples")
    mechanism, residuals, degradation, timings, visual_index = [], [], [], [], []
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for index, batch in enumerate(loader, 1):
            rgb, thermal, depth, _, _, (height, width), name = batch
            rgb, thermal, depth = (x.cuda(non_blocking=True).float() for x in (rgb, thermal, depth))
            torch.cuda.synchronize(); tic = time.perf_counter()
            prob, aux = predict(model, qnet, rgb, thermal, depth, height[0], width[0])
            pred = prob.squeeze().cpu().numpy(); pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            stem = Path(name[0]).stem; out_path = a.output / (stem + ".png")
            cv2.imwrite(str(out_path), np.rint(pred * 255).astype(np.uint8))
            torch.cuda.synchronize(); timings.append({"image": stem, "seconds": time.perf_counter() - tic})
            arrays = {}
            for scale, item in enumerate(aux["scales"]):
                if a.mode == "CCC_ONLY":
                    for n in ("Avd", "Avt", "Adt", "Rv", "Rd", "Rt"):
                        tensor = item["ccc"][n]; arrays["scale{}_{}".format(scale, n)] = tensor.cpu().numpy()[0, 0]
                        mechanism.append({"image": stem, "scale": scale, "tensor": n, **stats(tensor)})
                    cap = ccc_capture[scale]
                    for token, raw, cal in zip(("v", "d", "t"), cap["raw"], cap["cal"]):
                        delta = cal - raw; dn, bn, ratio = norm_ratio(delta, raw)
                        residuals.append({"image": stem, "mode": a.mode, "scale": scale,
                                          "branch": token, "delta_norm": dn, "base_norm": bn,
                                          "norm_ratio": ratio, "cal_mean": float(cal.mean()),
                                          "cal_std": float(cal.float().std(unbiased=False))})
                else:
                    rtf = item["rtf"]
                    error = float((rtf["W"].sum(1) - 1).abs().max())
                    if error >= 1e-5: raise RuntimeError("Softmax error {}".format(error))
                    w = rtf["W"].float(); entropy = -(w * torch.log(w.clamp_min(1e-12))).sum(1)
                    dominant = w.argmax(1)
                    for ci, n in enumerate(("Wv", "Wd", "Wt")):
                        tensor = rtf[n]; arrays["scale{}_{}".format(scale, n)] = tensor.cpu().numpy()[0, 0]
                        row = {"image": stem, "scale": scale, "tensor": n, **stats(tensor),
                               "spatial_variance": float(tensor.float().var(unbiased=False)),
                               "dominant_area_ratio": float((dominant == ci).float().mean()),
                               "entropy_mean": float(entropy.mean()), "softmax_sum_error": error,
                               "saturated_low_ratio": float((tensor < 1e-4).float().mean()),
                               "saturated_high_ratio": float((tensor > .9999).float().mean())}
                        mechanism.append(row)
                    for n in ("Qv", "Qd", "Qt"):
                        tensor = rtf[n]; arrays["scale{}_{}".format(scale, n)] = tensor.cpu().numpy()[0, 0]
                        mechanism.append({"image": stem, "scale": scale, "tensor": n, **stats(tensor)})
                    for branch, base, new in (("d", item["X_d_base"], item["X_d_new"]),
                                              ("t", item["X_t_base"], item["X_t_new"])):
                        dn, bn, ratio = norm_ratio(new - base, base)
                        residuals.append({"image": stem, "mode": a.mode, "scale": scale,
                                          "branch": branch, "delta_norm": dn, "base_norm": bn,
                                          "norm_ratio": ratio})
            if index <= a.artifact_count:
                np.savez_compressed(raw_dir / (stem + ".npz"), **arrays)
                for key, arr in arrays.items():
                    cv2.imwrite(str(map_dir / (stem + "_" + key + ".png")), np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8))
                panels = [panel(test_root/"V"/(stem+".png"), "RGB", True),
                          panel(test_root/"D"/(stem+".png"), "Depth"),
                          panel(test_root/"T"/(stem+".png"), "Thermal"),
                          panel(test_root/"GT"/(stem+".png"), "GT"),
                          panel(a.b0_pred/(stem+".png"), "B0"),
                          panel(a.full_pred/(stem+".png"), "FULL"),
                          panel(out_path, a.mode)]
                blank = np.zeros_like(panels[0])
                canvas = np.vstack((np.hstack(panels[:4]), np.hstack(panels[4:] + [blank])))
                vpath = visual_dir / (stem + "_comparison.png"); cv2.imwrite(str(vpath), canvas)
                visual_index.append({"image": stem, "mode": a.mode, "visual": str(vpath),
                                     "raw_tensor": str(raw_dir/(stem+".npz"))})
                if a.mode == "RTF_ONLY":
                    base_weights = {(s, n): float(item["rtf"][n].mean()) for s, item in enumerate(aux["scales"]) for n in ("Wv", "Wd", "Wt")}
                    base_prob = prob
                    variants = {"rgb_zero": (torch.zeros_like(rgb), thermal, depth),
                                "depth_zero": (rgb, thermal, torch.zeros_like(depth)),
                                "thermal_zero": (rgb, torch.zeros_like(thermal), depth)}
                    for kind, (rv, tv, dv) in variants.items():
                        degraded_prob, degraded_aux = predict(model, qnet, rv, tv, dv, height[0], width[0])
                        pdiff = float((degraded_prob - base_prob).abs().mean())
                        for scale, it in enumerate(degraded_aux["scales"]):
                            for n in ("Wv", "Wd", "Wt"):
                                before, after = base_weights[(scale, n)], float(it["rtf"][n].mean())
                                degradation.append({"image": stem, "diagnostic": kind, "scale": scale,
                                                    "weight": n, "before_mean": before, "after_mean": after,
                                                    "weight_delta": after-before, "prediction_mean_abs_delta": pdiff,
                                                    "interpretation_limit": "normalized tensor zeroing; not physical sensor failure"})
            print("{}/1000 {}".format(index, stem), flush=True)
    for h in handles: h.remove()
    preds = list(a.output.glob("*.png"))
    if len(preds) != 1000: raise RuntimeError("prediction count {}".format(len(preds)))
    gt_stems = {p.stem for p in (test_root/"GT").glob("*.png")}
    if {p.stem for p in preds} != gt_stems: raise RuntimeError("Prediction/GT name mismatch")
    constant = []
    for p in preds:
        x = cv2.imread(str(p), 0)
        if x is None: raise RuntimeError("Unreadable prediction {}".format(p))
        if int(x.max()) == int(x.min()): constant.append(p.stem)
    if constant: raise RuntimeError("Constant predictions: {}".format(constant[:10]))
    write_csv(a.artifact_dir / "mechanism_statistics.csv", mechanism)
    write_csv(a.artifact_dir / "residual_norm_statistics.csv", residuals)
    if degradation: write_csv(a.artifact_dir / "degradation_diagnostics.csv", degradation)
    write_csv(a.artifact_dir / "visualization_index.csv", visual_index)
    write_csv(a.output / "inference_times.csv", timings)
    manifest = {"status": "COMPLETE", "mode": a.mode, "count": len(preds),
                "seconds": time.perf_counter()-started,
                "mean_latency_seconds": float(np.mean([r["seconds"] for r in timings])),
                "fps": 1.0/float(np.mean([r["seconds"] for r in timings])),
                "peak_vram_mib": torch.cuda.max_memory_allocated()/2**20,
                "constant_predictions": len(constant), "qanet_sha256": sha256(a.qanet),
                "model_sha256": sha256(a.model), "qanet_load": q_report, "model_load": m_report,
                "raw_tensor_sets": len(list(raw_dir.glob("*.npz"))),
                "visual_candidates": len(list(visual_dir.glob("*.png"))),
                "postprocess": "Round1/Round2 sigmoid + bilinear align_corners=True + per-image min-max + uint8 PNG"}
    write_json(a.output / "inference_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__": main()
