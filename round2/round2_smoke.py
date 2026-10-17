from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from round2_common import (
    ROUND2_SEED,
    atomic_save,
    full_loss,
    gradient_norm,
    load_round1_tnet,
    load_strict,
    parameter_change,
    parameter_snapshot,
    seed_all,
    seed_worker,
    tensor_stats,
    validate_train_data,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--qanet", type=Path, required=True)
    parser.add_argument("--tnet", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=ROUND2_SEED)
    return parser.parse_args()


def to_cuda(batch):
    rgb, thermal, depth, edge, label = batch[:5]
    return tuple(x.cuda(non_blocking=True).float() for x in (rgb, thermal, depth, edge, label))


def make_loader(Data, train_root, batch_size, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Data(str(train_root)), batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        worker_init_fn=seed_worker, generator=generator,
    )


def named_grad_stats(model, token):
    squared, tensors, nonzero = 0.0, 0, 0
    for name, parameter in model.named_parameters():
        if token not in name or parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm())
        squared += norm * norm
        tensors += 1
        nonzero += int(norm > 0)
    return {"norm": squared ** 0.5, "tensors": tensors, "nonzero": nonzero}


def inspect_aux(mode, batch_size, aux, shape_rows, evidence_rows):
    for scale, item in enumerate(aux["scales"]):
        for name, shape in item["input_shapes"].items():
            shape_rows.append({
                "mode": mode, "batch_size": batch_size, "scale": scale,
                "tensor": name, "shape": "x".join(map(str, shape)),
            })
        ccc = item["ccc"]
        if ccc is not None:
            for name in ("Avd", "Avt", "Adt", "Rv", "Rd", "Rt"):
                stats = tensor_stats(ccc[name])
                if not stats["finite"] or stats["min"] < -1e-6 or stats["max"] > 1 + 1e-6:
                    raise RuntimeError("Invalid CCC tensor {} at scale {}: {}".format(name, scale, stats))
                evidence_rows.append({"mode": mode, "batch_size": batch_size, "scale": scale,
                                      "branch": "CCC", "tensor": name, **stats})
        rtf = item["rtf"]
        if rtf is not None:
            softmax_error = float((rtf["W"].sum(dim=1) - 1).abs().max())
            if softmax_error >= 1e-5:
                raise RuntimeError("RTF Softmax error {} exceeds 1e-5".format(softmax_error))
            for name in ("Wv", "Wd", "Wt"):
                stats = tensor_stats(rtf[name])
                if not stats["finite"] or stats["min"] < -1e-6 or stats["max"] > 1 + 1e-6:
                    raise RuntimeError("Invalid RTF tensor {} at scale {}: {}".format(name, scale, stats))
                evidence_rows.append({"mode": mode, "batch_size": batch_size, "scale": scale,
                                      "branch": "RTF", "tensor": name,
                                      "softmax_max_abs_error": softmax_error, **stats})


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "round2"))
    from lib.dataset import Data
    from QAnet import QAnet
    from round2.RACFNet import RACFNet

    train_root = validate_train_data(args.data_root)
    out = args.work_root.resolve() / "round2" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    shape_rows, smoke_rows, evidence_rows, complexity_rows = [], [], [], []
    load_reports = {}

    for mode_index, mode in enumerate(RACFNet.MODES):
        seed_all(args.seed + mode_index)
        model = RACFNet(mode)
        load_reports[mode] = load_round1_tnet(model, args.tnet)
        qnet = QAnet()
        load_reports[mode + "_QANET"] = load_strict(qnet, args.qanet, "Round1 QAnet")
        model.cuda().train()
        qnet.cuda().train()
        for parameter in qnet.parameters():
            parameter.requires_grad_(False)

        loaders = {
            1: make_loader(Data, train_root, 1, min(args.workers, 2), args.seed + 11),
            4: make_loader(Data, train_root, 4, args.workers, args.seed + 44),
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fixed_cpu = None
        for batch_size, loader in loaders.items():
            batch = next(iter(loader))
            if batch_size == 1:
                fixed_cpu = tuple(x[:1].cpu() for x in batch[:5])
            rgb, thermal, depth, edge, label = to_cuda(batch)
            with torch.no_grad():
                dlogit, tlogit = qnet(rgb, thermal, depth)
                outputs, aux = model(rgb, thermal, depth, dlogit, tlogit, return_aux=True)
            if len(outputs) != 14:
                raise RuntimeError("{} returned {} outputs".format(mode, len(outputs)))
            for index, prediction in enumerate(outputs):
                if prediction.shape != label.shape:
                    raise RuntimeError(
                        "{} output {} shape {} != label {}".format(mode, index, prediction.shape, label.shape)
                    )
                stats = tensor_stats(prediction)
                if not stats["finite"]:
                    raise RuntimeError("Non-finite prediction in {} output {}".format(mode, index))
                shape_rows.append({"mode": mode, "batch_size": batch_size, "scale": -1,
                                   "tensor": "output_{}".format(index),
                                   "shape": "x".join(map(str, prediction.shape))})
            inspect_aux(mode, batch_size, aux, shape_rows, evidence_rows)
            del outputs, aux, dlogit, tlogit, rgb, thermal, depth, edge, label, batch

        model.eval()
        qnet.eval()
        with torch.no_grad():
            rgb, thermal, depth, _, _ = (x.cuda().float() for x in fixed_cpu)
            dlogit, tlogit = qnet(rgb, thermal, depth)
            prediction_before = model(rgb, thermal, depth, dlogit, tlogit)[-1].detach().cpu()
        model.train()
        qnet.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        prefixes = ("ccc_scales.", "rtf_scales.", "rtf_quality_v.")
        new_before = parameter_snapshot(model, prefixes)
        train_loader = make_loader(Data, train_root, 4, args.workers, args.seed + 100 + mode_index)
        losses, step_times = [], []
        first_gradients = None
        for step, batch in enumerate(train_loader, 1):
            if step > args.steps:
                break
            rgb, thermal, depth, edge, label = to_cuda(batch)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                dlogit, tlogit = qnet(rgb, thermal, depth)
            outputs = model(rgb, thermal, depth, dlogit, tlogit)
            loss = full_loss(outputs, edge, label)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite smoke loss for {} step {}".format(mode, step))
            loss.backward()
            ccc_grad = gradient_norm(model.named_parameters(), "ccc_scales.")
            rtf_grad = gradient_norm(model.named_parameters(), "rtf_scales.")
            consistency_grad = named_grad_stats(model, ".consistency_")
            self_grad = named_grad_stats(model, ".self_")
            if step == 1:
                first_gradients = {
                    "ccc_norm": ccc_grad[0], "ccc_tensors": ccc_grad[1], "ccc_nonzero": ccc_grad[2],
                    "rtf_norm": rtf_grad[0], "rtf_tensors": rtf_grad[1], "rtf_nonzero": rtf_grad[2],
                    "consistency": consistency_grad, "self_evidence": self_grad,
                }
                if mode in ("CCC_ONLY", "FULL"):
                    if ccc_grad[0] <= 0 or consistency_grad["nonzero"] == 0 or self_grad["nonzero"] == 0:
                        raise RuntimeError("CCC gradient validation failed for {}".format(mode))
                if mode in ("RTF_ONLY", "FULL") and rtf_grad[0] <= 0:
                    raise RuntimeError("RTF gradient validation failed for {}".format(mode))
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            value = float(loss.detach())
            losses.append(value)
            step_times.append(seconds)
            print(json.dumps({"mode": mode, "step": step, "loss": value,
                              "step_seconds": seconds}), flush=True)
            del outputs, loss, dlogit, tlogit, rgb, thermal, depth, edge, label, batch

        changes = parameter_change(new_before, model)
        if changes["changed_parameter_tensors"] == 0 or changes["sum_abs_delta"] <= 0:
            raise RuntimeError("New module parameters did not update for {}: {}".format(mode, changes))
        model.eval()
        qnet.eval()
        with torch.no_grad():
            rgb, thermal, depth, _, _ = (x.cuda().float() for x in fixed_cpu)
            dlogit, tlogit = qnet(rgb, thermal, depth)
            prediction_after = model(rgb, thermal, depth, dlogit, tlogit)[-1].detach().cpu()
        prediction_delta = float((prediction_after - prediction_before).abs().mean())
        if prediction_delta <= 0:
            raise RuntimeError("Final prediction was not affected by optimization for {}".format(mode))

        checkpoint = out / "{}_30step.pth".format(mode.lower())
        atomic_save(model.state_dict(), checkpoint)
        verifier = RACFNet(mode)
        reload_report = load_strict(verifier, checkpoint, "{} smoke strict reload".format(mode))
        result = {
            "mode": mode,
            "status": "PASS",
            "steps": len(losses),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "mean_loss": float(np.mean(losses)),
            "mean_step_seconds": float(np.mean(step_times)),
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
            "prediction_mean_abs_change": prediction_delta,
            "first_step_gradients": first_gradients,
            "new_parameter_change": changes,
            "checkpoint_reload": reload_report,
        }
        write_json(out / "{}_result.json".format(mode.lower()), result)
        smoke_rows.append({
            "mode": mode, "status": "PASS", "steps": len(losses),
            "initial_loss": losses[0], "final_loss": losses[-1],
            "mean_loss": float(np.mean(losses)), "mean_step_seconds": float(np.mean(step_times)),
            "peak_vram_mib": result["peak_vram_mib"],
            "ccc_grad_norm": first_gradients["ccc_norm"],
            "rtf_grad_norm": first_gradients["rtf_norm"],
            "changed_new_tensors": changes["changed_parameter_tensors"],
            "tracked_new_tensors": changes["tracked_parameter_tensors"],
            "prediction_mean_abs_change": prediction_delta,
        })
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        new_parameters = sum(parameter.numel() for name, parameter in model.named_parameters()
                             if name.startswith(prefixes))
        complexity_rows.append({
            "mode": mode, "parameters": total_parameters, "new_parameters": new_parameters,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "batch_size": 4, "input_height": 384, "input_width": 384,
            "mean_train_step_seconds": float(np.mean(step_times)),
            "peak_train_vram_mib": result["peak_vram_mib"],
        })
        del verifier, model, qnet, optimizer, loaders, train_loader
        torch.cuda.empty_cache()

    write_csv(out / "ROUND2_TENSOR_SHAPES.csv", shape_rows)
    write_csv(out / "ROUND2_SMOKE_RESULTS.csv", smoke_rows)
    write_csv(out / "ROUND2_SMOKE_EVIDENCE_STATS.csv", evidence_rows)
    write_csv(out / "ROUND2_COMPLEXITY.csv", complexity_rows)
    write_json(out / "ROUND2_CHECKPOINT_LOAD_AUDIT.json", load_reports)
    print(json.dumps({"status": "PASS", "modes": [row["mode"] for row in smoke_rows]}, indent=2))


if __name__ == "__main__":
    main()
