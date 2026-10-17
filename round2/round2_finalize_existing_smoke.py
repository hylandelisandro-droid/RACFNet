from __future__ import annotations

import csv
import gc
import json
import sys
from pathlib import Path

from round2_common import write_csv, write_json


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: round2_finalize_existing_smoke.py REPO WORK_ROOT")
    repo = Path(sys.argv[1]).resolve()
    work = Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(repo))
    from round2.RACFNet import RACFNet

    smoke = work / "round2" / "smoke"
    results_csv = smoke / "ROUND2_SMOKE_RESULTS.csv"
    with results_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if {row["mode"] for row in rows} != set(RACFNet.MODES):
        raise RuntimeError("Smoke CSV does not contain all three modes")
    if any(row["status"] != "PASS" or int(row["steps"]) != 30 for row in rows):
        raise RuntimeError("Smoke CSV contains a failed or incomplete mode")

    audit = {"status": "PASS", "note": "Recovered final export after CSV schema-only failure", "modes": {}}
    complexity = []
    for mode in RACFNet.MODES:
        stem = mode.lower()
        result_path = smoke / (stem + "_result.json")
        checkpoint = smoke / (stem + "_30step.pth")
        if not result_path.is_file() or not checkpoint.is_file():
            raise RuntimeError("Missing completed smoke artifacts for {}".format(mode))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        reload_report = result["checkpoint_reload"]
        if result["status"] != "PASS" or not reload_report["strict"]:
            raise RuntimeError("Smoke result or strict reload failed for {}".format(mode))
        if reload_report["missing_keys"] or reload_report["unexpected_keys"]:
            raise RuntimeError("Checkpoint key mismatch for {}".format(mode))
        model = RACFNet(mode)
        prefixes = ("ccc_scales.", "rtf_scales.", "rtf_quality_v.")
        complexity.append({
            "mode": mode,
            "parameters": sum(p.numel() for p in model.parameters()),
            "new_parameters": sum(p.numel() for name, p in model.named_parameters()
                                  if name.startswith(prefixes)),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "batch_size": 4, "input_height": 384, "input_width": 384,
            "mean_train_step_seconds": result["mean_step_seconds"],
            "peak_train_vram_mib": result["peak_vram_mib"],
        })
        audit["modes"][mode] = {
            "steps": result["steps"],
            "checkpoint": str(checkpoint),
            "strict_reload": True,
            "new_parameter_change": result["new_parameter_change"],
            "first_step_gradients": result["first_step_gradients"],
            "prediction_mean_abs_change": result["prediction_mean_abs_change"],
        }
        del model
        gc.collect()

    partial = smoke / "ROUND2_SMOKE_EVIDENCE_STATS.csv"
    if partial.is_file():
        partial.replace(smoke / "ROUND2_SMOKE_EVIDENCE_STATS.partial.csv")
    write_csv(smoke / "ROUND2_COMPLEXITY.csv", complexity)
    write_json(smoke / "ROUND2_CHECKPOINT_LOAD_AUDIT.json", audit)
    (smoke / "SMOKE_VALIDATED").write_text(
        "All CCC_ONLY, RTF_ONLY, and FULL 30-step tests passed; final CSV export recovered.\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
