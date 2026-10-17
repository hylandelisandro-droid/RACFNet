from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from round2_common import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_root.resolve()
    repo = args.repo.resolve()
    round2 = work / "round2"
    results = round2 / "paper_results"
    results.mkdir(parents=True, exist_ok=True)

    copies = {
        round2 / "smoke" / "ROUND2_TENSOR_SHAPES.csv": results / "ROUND2_TENSOR_SHAPES.csv",
        round2 / "smoke" / "ROUND2_SMOKE_RESULTS.csv": results / "ROUND2_SMOKE_RESULTS.csv",
        round2 / "smoke" / "ROUND2_COMPLEXITY.csv": results / "ROUND2_COMPLEXITY.csv",
        round2 / "reliability_artifacts" / "ROUND2_RELIABILITY_STATISTICS.csv": results / "ROUND2_RELIABILITY_STATISTICS.csv",
        round2 / "reliability_artifacts" / "ROUND2_DEGRADATION_WEIGHT_CHANGES.csv": results / "ROUND2_DEGRADATION_WEIGHT_CHANGES.csv",
        round2 / "experiments" / "full" / "ROUND2_FULL_TRAINING_REPORT.md": results / "ROUND2_FULL_TRAINING_REPORT.md",
    }
    for source, destination in copies.items():
        if not source.is_file():
            raise RuntimeError("Missing required Round2 artifact: {}".format(source))
        shutil.copy2(source, destination)
    for source in (repo / "round2" / "docs").glob("ROUND2_*.md"):
        shutil.copy2(source, results / source.name)

    result = json.loads((round2 / "experiments" / "full" / "result.json").read_text(encoding="utf-8"))
    transfer = json.loads((round2 / "experiments" / "full" / "checkpoint_load_report.json").read_text(encoding="utf-8"))
    checkpoint = round2 / "checkpoints" / "RACFNet_FULL_final.pth"
    manifest = json.loads((round2 / "predictions" / "full" / "inference_manifest.json").read_text(encoding="utf-8"))
    metrics_path = results / "ROUND2_FULL_EIGHT_METRICS.csv"
    audit = """# Round2 checkpoint audit

- Final checkpoint: `{checkpoint}`
- SHA-256: `{digest}`
- Strict reload: `{strict}`
- Missing keys: `{missing}`
- Unexpected keys: `{unexpected}`
- Round1 Tnet keys all reused: `{reused}`
- Round1 QAnet loaded strictly: `{q_strict}`
- Formal prediction count: `{count}`
- Raw reliability tensor sets: `{raw}`
- Visual candidates: `{visual}`
- Formal metrics file: `{metrics}`
""".format(
        checkpoint=checkpoint,
        digest=sha256(checkpoint),
        strict=result["final_reload"]["strict"],
        missing=result["final_reload"]["missing_keys"],
        unexpected=result["final_reload"]["unexpected_keys"],
        reused=transfer["tnet_transfer"]["all_round1_keys_reused"],
        q_strict=transfer["qanet"]["strict"],
        count=manifest["count"], raw=manifest["raw_tensor_sets"],
        visual=manifest["visual_candidates"], metrics=metrics_path,
    )
    (results / "ROUND2_CHECKPOINT_AUDIT.md").write_text(audit, encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "paper_results": str(results)}, indent=2))


if __name__ == "__main__":
    main()
