from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--qanet", type=Path, required=True)
    p.add_argument("--tnet", type=Path, required=True)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20261013)
    return p.parse_args()


def loader(Data, train_root, batch, workers, seed, seed_worker):
    return DataLoader(Data(str(train_root)), batch_size=batch, shuffle=True, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed))


def cuda_batch(batch):
    return tuple(x.cuda(non_blocking=True).float() for x in batch[:5])


def main():
    a = args_parser()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    repo = a.repo.resolve(); sys.path[:0] = [str(repo), str(repo / "round2")]
    from lib.dataset import Data
    from QAnet import QAnet
    from round2.RACFNet import RACFNet
    from round2.round2_common import (atomic_save, full_loss, gradient_norm, load_round1_tnet,
        load_strict, parameter_change, parameter_snapshot, seed_all, seed_worker, tensor_stats,
        validate_train_data, write_csv, write_json)
    train_root = validate_train_data(a.data_root)
    out = a.work_root.resolve() / "round3" / "smoke"; out.mkdir(parents=True, exist_ok=True)
    rows, shapes, loads = [], [], {}
    for offset, mode in enumerate(("RTF_ONLY", "CCC_ONLY")):
        seed_all(a.seed + offset)
        model, qnet = RACFNet(mode), QAnet()
        loads[mode] = load_round1_tnet(model, a.tnet)
        loads[mode + "_QANET"] = load_strict(qnet, a.qanet, "Round1 QAnet")
        model.cuda().train(); qnet.cuda().train()
        for p in qnet.parameters(): p.requires_grad_(False)
        fixed = None
        for batch_size in (1, 4):
            batch = next(iter(loader(Data, train_root, batch_size, min(a.workers, 2), a.seed + batch_size, seed_worker)))
            if batch_size == 1: fixed = tuple(x.cpu() for x in batch[:5])
            rgb, thermal, depth, _, label = cuda_batch(batch)
            with torch.no_grad():
                dl, tl = qnet(rgb, thermal, depth)
                outputs, aux = model(rgb, thermal, depth, dl, tl, return_aux=True)
            if len(outputs) != 14: raise RuntimeError("{} output count {}".format(mode, len(outputs)))
            for idx, pred in enumerate(outputs):
                if pred.shape != label.shape or not tensor_stats(pred)["finite"]:
                    raise RuntimeError("{} bad output {}".format(mode, idx))
                shapes.append({"mode": mode, "batch_size": batch_size, "tensor": "output_{}".format(idx),
                               "shape": "x".join(map(str, pred.shape))})
            for scale, item in enumerate(aux["scales"]):
                if tuple(item["input_shapes"]["Fv"][1:]) != (128, 96, 96):
                    raise RuntimeError("Unexpected integration shape {}".format(item["input_shapes"]))
                if mode == "CCC_ONLY":
                    if item["ccc"] is None or item["rtf"] is not None: raise RuntimeError("CCC_ONLY switch leak")
                    for name in ("Avd", "Avt", "Adt", "Rv", "Rd", "Rt"):
                        s = tensor_stats(item["ccc"][name])
                        if not s["finite"] or s["min"] < -1e-6 or s["max"] > 1.000001:
                            raise RuntimeError("bad CCC evidence")
                else:
                    if item["ccc"] is not None or item["rtf"] is None: raise RuntimeError("RTF_ONLY switch leak")
                    err = float((item["rtf"]["W"].sum(1) - 1).abs().max())
                    if err >= 1e-5: raise RuntimeError("softmax error {}".format(err))
                    for name in ("Qv", "Qd", "Qt", "Wv", "Wd", "Wt"):
                        if not tensor_stats(item["rtf"][name])["finite"]: raise RuntimeError("bad RTF evidence")
        model.eval(); qnet.eval()
        with torch.no_grad():
            rgb, thermal, depth, _, _ = (x.cuda().float() for x in fixed)
            dl, tl = qnet(rgb, thermal, depth)
            before_pred = model(rgb, thermal, depth, dl, tl)[-1].detach().cpu()
        model.train(); qnet.train()
        prefixes = ("ccc_scales.",) if mode == "CCC_ONLY" else ("rtf_scales.", "rtf_quality_v.")
        snap = parameter_snapshot(model, prefixes)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.5, 0.999))
        losses, times, first_grad = [], [], None
        torch.cuda.reset_peak_memory_stats()
        train_loader = loader(Data, train_root, 4, a.workers, a.seed + 100 + offset, seed_worker)
        for step, batch in enumerate(train_loader, 1):
            if step > a.steps: break
            rgb, thermal, depth, edge, label = cuda_batch(batch)
            opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); tic = time.perf_counter()
            with torch.no_grad(): dl, tl = qnet(rgb, thermal, depth)
            outputs = model(rgb, thermal, depth, dl, tl); loss = full_loss(outputs, edge, label)
            if not torch.isfinite(loss): raise RuntimeError("{} nonfinite loss".format(mode))
            loss.backward()
            if first_grad is None:
                ccc = gradient_norm(model.named_parameters(), "ccc_scales.")
                rtf = gradient_norm(model.named_parameters(), "rtf_scales.")
                first_grad = {"ccc": ccc, "rtf": rtf}
                active = ccc[0] if mode == "CCC_ONLY" else rtf[0]
                if active <= 0: raise RuntimeError("{} active gradient is zero".format(mode))
            opt.step(); torch.cuda.synchronize()
            losses.append(float(loss)); times.append(time.perf_counter() - tic)
            print(json.dumps({"mode": mode, "step": step, "loss": losses[-1]}), flush=True)
        change = parameter_change(snap, model)
        if change["changed_parameter_tensors"] == 0 or change["sum_abs_delta"] <= 0:
            raise RuntimeError("{} active module parameters did not update".format(mode))
        model.eval(); qnet.eval()
        with torch.no_grad():
            rgb, thermal, depth, _, _ = (x.cuda().float() for x in fixed)
            dl, tl = qnet(rgb, thermal, depth)
            after_pred = model(rgb, thermal, depth, dl, tl)[-1].detach().cpu()
        pred_delta = float((after_pred - before_pred).abs().mean())
        if pred_delta <= 0: raise RuntimeError("{} prediction unchanged".format(mode))
        ckpt = out / "{}_30step.pth".format(mode.lower()); atomic_save(model.state_dict(), ckpt)
        strict = load_strict(RACFNet(mode), ckpt, mode + " smoke strict reload")
        result = {"mode": mode, "status": "PASS", "steps": len(losses),
                  "initial_loss": losses[0], "final_loss": losses[-1],
                  "mean_step_seconds": float(np.mean(times)),
                  "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "prediction_mean_abs_change": pred_delta, "first_gradients": first_grad,
                  "new_parameter_change": change, "strict_reload": strict,
                  "formal_training_must_reinitialize_from_round1": True}
        write_json(out / "{}_result.json".format(mode.lower()), result)
        rows.append({k: result[k] for k in ("mode", "status", "steps", "initial_loss", "final_loss",
                                            "mean_step_seconds", "peak_vram_mib", "prediction_mean_abs_change")})
        del model, qnet, opt, train_loader; torch.cuda.empty_cache()
    write_csv(out / "ROUND3_SMOKE_RESULTS.csv", rows)
    write_csv(out / "ROUND3_TENSOR_SHAPES.csv", shapes)
    write_json(out / "ROUND3_CHECKPOINT_LOAD_AUDIT.json", loads)
    (out / "SMOKE_VALIDATED").write_text("RTF_ONLY and CCC_ONLY fresh 30-step smoke passed\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "modes": [r["mode"] for r in rows]}), flush=True)


if __name__ == "__main__": main()
