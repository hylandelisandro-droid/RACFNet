from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from round2_common import (
    ROUND2_SEED,
    atomic_save,
    bn_diff,
    bn_snapshot,
    full_loss,
    load_round1_tnet,
    load_strict,
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=ROUND2_SEED)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    return parser.parse_args()


def reliability_rows(epoch, step, aux):
    rows = []
    for scale, item in enumerate(aux["scales"]):
        for branch, names, source in (
            ("CCC_CONSISTENCY", ("Avd", "Avt", "Adt"), item["ccc"]),
            ("CCC_RELIABILITY", ("Rv", "Rd", "Rt"), item["ccc"]),
            ("RTF_WEIGHT", ("Wv", "Wd", "Wt"), item["rtf"]),
        ):
            for name in names:
                rows.append({"epoch": epoch, "step": step, "scale": scale,
                             "branch": branch, "tensor": name, **tensor_stats(source[name])})
        rows.append({
            "epoch": epoch, "step": step, "scale": scale,
            "branch": "RTF_SOFTMAX", "tensor": "sum_error",
            "min": 0.0,
            "max": float((item["rtf"]["W"].sum(dim=1) - 1).abs().max()),
            "mean": 0.0, "std": 0.0, "finite": True,
        })
    return rows


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    repo = args.repo.resolve()
    work = args.work_root.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "round2"))
    from lib.dataset import Data
    from QAnet import QAnet
    from round2.RACFNet import RACFNet

    train_root = validate_train_data(args.data_root)
    run_dir = work / "round2" / "experiments" / "full"
    checkpoint_dir = work / "round2" / "checkpoints" / "periodic"
    final_dir = work / "round2" / "checkpoints"
    for folder in (run_dir, checkpoint_dir, final_dir):
        folder.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)

    model = RACFNet("FULL")
    qnet = QAnet()
    reports = {
        "tnet_transfer": load_round1_tnet(model, args.tnet),
        "qanet": load_strict(qnet, args.qanet, "Round1 QAnet"),
    }
    model.cuda().train()
    qnet.cuda().train()
    for parameter in qnet.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    start_epoch = 1
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume["model"], strict=True)
        qnet.load_state_dict(resume["qnet"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(resume["epoch"]) + 1
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"])
        torch.cuda.set_rng_state_all(resume["cuda_rng"])

    original_parameters = sum(p.numel() for name, p in model.named_parameters()
                              if not name.startswith(("ccc_scales.", "rtf_scales.")))
    ccc_parameters = sum(p.numel() for name, p in model.named_parameters()
                         if name.startswith("ccc_scales."))
    rtf_parameters = sum(p.numel() for name, p in model.named_parameters()
                         if name.startswith("rtf_scales."))
    config = {
        "project": "RACFNet", "round": 2, "mode": "FULL",
        "repo": str(repo), "data_root": str(args.data_root.resolve()), "work_root": str(work),
        "epochs": args.epochs, "batch_size": args.batch_size, "workers": args.workers,
        "optimizer": "Adam", "lr_original": args.lr, "lr_new_modules": args.lr,
        "betas": [0.5, 0.999], "precision": "FP32", "seed": args.seed,
        "initialization": "Round1 Tnet_final.pth", "qanet": "Round1 QAnet_final.pth",
        "semantic_mode": "B0 original quality path + sigmoid evidence inside CCC/RTF",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {"original": original_parameters, "ccc": ccc_parameters,
                       "rtf": rtf_parameters, "total": sum(p.numel() for p in model.parameters())},
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "checkpoint_load_report.json", reports)

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        Data(str(train_root)), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker, generator=generator,
    )
    rows, reliability = [], []
    qnet_bn_before = bn_snapshot(qnet)
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        epoch_losses, epoch_times = [], []
        for step, batch in enumerate(loader, 1):
            rgb, thermal, depth, edge, label = (
                tensor.cuda(non_blocking=True).float() for tensor in batch[:5]
            )
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            with torch.no_grad():
                dlogit, tlogit = qnet(rgb, thermal, depth)
            if step == 1:
                outputs, aux = model(rgb, thermal, depth, dlogit, tlogit, return_aux=True)
                reliability.extend(reliability_rows(epoch, step, aux))
                del aux
            else:
                outputs = model(rgb, thermal, depth, dlogit, tlogit)
            loss = full_loss(outputs, edge, label)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss at epoch={}, step={}".format(epoch, step))
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - step_started
            value = float(loss.detach())
            epoch_losses.append(value)
            epoch_times.append(seconds)
            row = {
                "epoch": epoch, "step": step, "loss": value,
                "step_seconds": seconds,
                "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        summary = {
            "epoch_summary": epoch,
            "mean_loss": float(np.mean(epoch_losses)),
            "seconds": time.perf_counter() - epoch_started,
            "mean_step_seconds": float(np.mean(epoch_times)),
            "steps": len(epoch_losses),
        }
        print(json.dumps(summary), flush=True)
        write_csv(run_dir / "loss_steps.csv", rows)
        write_csv(run_dir / "reliability_training_statistics.csv", reliability)
        if epoch % 10 == 0:
            free_gib = shutil.disk_usage(work).free / 1024**3
            if free_gib < args.min_free_gb:
                raise RuntimeError("Disk guard: only {:.2f} GiB free".format(free_gib))
            atomic_save(model.state_dict(), checkpoint_dir / "racfnet_full_epoch_{:03d}.pth".format(epoch))
            atomic_save({
                "epoch": epoch, "model": model.state_dict(), "qnet": qnet.state_dict(),
                "optimizer": optimizer.state_dict(), "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            }, checkpoint_dir / "resume_latest.pth")

    final_path = final_dir / "RACFNet_FULL_final.pth"
    atomic_save(model.state_dict(), final_path)
    verifier = RACFNet("FULL")
    final_reload = load_strict(verifier, final_path, "RACFNet FULL final strict reload")
    result = {
        "status": "COMPLETE", "mode": "FULL", "epochs_completed": args.epochs,
        "steps": len(rows), "wall_seconds": time.perf_counter() - training_started,
        "mean_step_seconds": float(np.mean([row["step_seconds"] for row in rows])),
        "estimated_epoch_seconds": float(np.mean([row["step_seconds"] for row in rows])) * len(loader),
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "free_disk_gib": shutil.disk_usage(work).free / 1024**3,
        "qanet_bn_train_mode_change": bn_diff(qnet_bn_before, bn_snapshot(qnet)),
        "final_checkpoint": str(final_path), "final_reload": final_reload,
        "parameters": config["parameters"],
    }
    write_json(run_dir / "result.json", result)
    (run_dir / "ROUND2_FULL_TRAINING_REPORT.md").write_text(
        "# Round2 FULL training report\n\n```json\n" +
        json.dumps(result, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
