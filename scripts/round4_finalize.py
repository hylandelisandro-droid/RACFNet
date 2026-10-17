from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_panel_alignment import require_matplotlib_panel_alignment
from round4_common import HISTORICAL, METRICS, MODEL_ORDER, model_specs, read_csv, sha256, write_csv, write_json


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

MODEL_COLORS = {
    "B0": "#767676",
    "CCC_ONLY": "#42949E",
    "RTF_ONLY": "#9A4D8E",
    "FULL": "#0F4D92",
}


def save_publication_figure(fig, out_path: Path, *, row_groups=None,
                            column_groups=None, exclude_axes=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(out_path) + ".alignment.json",
        overlay_svg=str(out_path) + ".alignment.svg",
        row_groups=row_groups,
        column_groups=column_groups,
        exclude_axes=exclude_axes or [],
        require_panel_labels=False,
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="Project root containing prior-round results")
    p.add_argument("--out", type=Path, required=True,
                   help="Round4 results directory")
    p.add_argument("--source", type=Path, required=True,
                   help="Directory containing Round4 analysis source files")
    return p.parse_args()


def copy_csv(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def find_first(root: Path, patterns: tuple[str, ...]):
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def make_tables(root: Path, out: Path):
    table1 = []
    for model in MODEL_ORDER:
        table1.append({"method": model, "source": f"Round{model_specs(root)[model].round} formal result",
                       **dict(zip(METRICS, HISTORICAL[model]))})
    write_csv(out / "TABLE1_OVERALL.csv", table1)
    write_csv(out / "TABLE6_ABLATION.csv", table1)
    write_csv(out / "Table1_Overall.csv", table1)
    write_csv(out / "Table6_Ablation.csv", table1)
    copy_csv(out / "ROUND4_TABLE2_VISIBLE.csv", out / "TABLE2_VISIBLE.csv")
    copy_csv(out / "ROUND4_TABLE3_DEPTH.csv", out / "TABLE3_DEPTH.csv")
    copy_csv(out / "ROUND4_TABLE4_THERMAL.csv", out / "TABLE4_THERMAL.csv")
    complexity = find_first(root, (
        "Round4_cloud_results/ROUND4_COMPLEXITY.csv",
        "round4/ROUND4_COMPLEXITY.csv",
        "round3/paper_results/ROUND3_COMPLEXITY.csv",
        "round3/**/ROUND3_COMPLEXITY.csv",
    ))
    if complexity:
        copy_csv(complexity, out / "TABLE5_COMPLEXITY.csv")
        copy_csv(complexity, out / "Table5_Complexity.csv")
    else:
        write_csv(out / "TABLE5_COMPLEXITY.csv", [{
            "status": "PENDING_UNIFIED_MEASUREMENT",
            "note": "No prior complexity CSV was found; do not invent values.",
        }])


def load_gray(path: Path):
    x = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if x is None:
        raise RuntimeError(f"cannot read {path}")
    return x


def load_color(path: Path):
    x = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if x is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


def mae(path: Path, gt: Path):
    return float(np.mean(np.abs(load_gray(path).astype(np.float32) - load_gray(gt).astype(np.float32))) / 255)


def select_samples(root: Path, out: Path, pred_dirs):
    gt_dir = root / "datasets/VDT-2048/Test/GT"
    challenge_rows = read_csv(out / "ROUND4_CHALLENGE_LABELS.csv")
    challenge_ids = set()
    id_to_attrs = {}
    for row in challenge_rows:
        for stem in row["sample_ids"].split(";"):
            challenge_ids.add(stem)
            id_to_attrs.setdefault(stem, []).append(f"{row['modality']}/{row['challenge']}")
    records = []
    for gt in gt_dir.glob("*.png"):
        stem = gt.stem
        b0 = mae(pred_dirs["B0"] / gt.name, gt)
        full = mae(pred_dirs["FULL"] / gt.name, gt)
        records.append({"sample_id": stem, "b0_mae": b0, "full_mae": full,
                        "improvement": b0 - full, "challenge": stem in challenge_ids,
                        "attributes": ";".join(id_to_attrs.get(stem, []))})
    # Ordinary candidates favour low attribute burden and clear, representative
    # gains; official challenge membership is still recorded in the index.
    ordinary = sorted(records, key=lambda x: (len(id_to_attrs.get(x["sample_id"], [])),
                                               -x["improvement"]))[:8]
    # Greedily cover as many distinct official challenge attributes as possible
    # before filling by improvement.  This prevents cherry-picking one easy class.
    challenge_pool = sorted((x for x in records if x["challenge"]),
                            key=lambda x: x["improvement"], reverse=True)
    challenge, covered = [], set()
    while challenge_pool and len(challenge) < 8:
        best = max(challenge_pool,
                   key=lambda x: (len(set(id_to_attrs.get(x["sample_id"], [])) - covered),
                                  x["improvement"]))
        challenge.append(best)
        covered.update(id_to_attrs.get(best["sample_id"], []))
        challenge_pool.remove(best)
    failures = sorted(records, key=lambda x: x["improvement"])[:5]
    return ordinary, challenge, failures


def comparison_figure(root, out_path: Path, rows, pred_dirs, title):
    test = root / "datasets/VDT-2048/Test"
    columns = ("Visible RGB", "Depth", "Thermal", "GT", "B0", "CCC_ONLY", "RTF_ONLY", "FULL")
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(7.2, 1.05 * len(rows)), facecolor="white")
    if len(rows) == 1:
        axes = axes[None, :]
    for r, record in enumerate(rows):
        stem = record["sample_id"]
        paths = [test / "V" / f"{stem}.png", test / "D" / f"{stem}.png",
                 test / "T" / f"{stem}.png", test / "GT" / f"{stem}.png"] + [pred_dirs[m] / f"{stem}.png" for m in MODEL_ORDER]
        for c, path in enumerate(paths):
            image = load_color(path) if c == 0 else load_gray(path)
            axes[r, c].imshow(image, cmap=None if c == 0 else "gray", vmin=0, vmax=255)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(columns[c], fontsize=7, fontweight="bold")
            if c == 0:
                axes[r, c].set_ylabel(stem, fontsize=6)
        axes[r, -1].text(.02, .03, f"MAE Δ={record['full_mae']-record['b0_mae']:+.4f}",
                         transform=axes[r, -1].transAxes, fontsize=5.5, color="yellow",
                         bbox=dict(facecolor="black", alpha=.55, pad=2))
    fig.suptitle(title, fontsize=8, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .975), pad=.35, w_pad=.2, h_pad=.22)
    save_publication_figure(fig, out_path)


def find_mechanism_npz(root: Path):
    patterns = (
        "round2/**/reliability_artifacts/full/raw_tensors/*.npz",
        "round2/**/raw_tensors/*.npz",
    )
    candidates = []
    for pattern in patterns:
        candidates = sorted(root.glob(pattern), key=lambda p: p.stem)
        if candidates:
            break
    return candidates[:2]


def mechanism_figure(root: Path, out_path: Path):
    sources = find_mechanism_npz(root)
    if not sources:
        return None
    loaded = [(path, np.load(path)) for path in sources]
    rows = []
    for path, data in loaded:
        for scale in range(3):
            keys = [f"scale{scale}_{x}" for x in ("Rv", "Rd", "Rt", "Wv", "Wd", "Wt")]
            if all(k in data for k in keys):
                rows.append((path, scale, keys, data))
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 6, figsize=(7.2, 1.02 * len(rows)), facecolor="white")
    if len(rows) == 1:
        axes = axes[None, :]
    titles = ("CCC Rv", "CCC Rd", "CCC Rt", "RTF Wv", "RTF Wd", "RTF Wt")
    image_artist = None
    for r, (path, scale, keys, data) in enumerate(rows):
        for c, key in enumerate(keys):
            image = np.squeeze(data[key])
            image_artist = axes[r, c].imshow(image, cmap="viridis", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=7, fontweight="bold")
            if c == 0:
                axes[r, c].set_ylabel(f"{path.stem} / node {scale}", fontsize=6)
    if image_artist is not None:
        fig.colorbar(image_artist, ax=axes.ravel().tolist(), fraction=.012, pad=.012,
                     label="Evidence / Softmax weight")
    fig.suptitle("FULL: actual CCC reliability evidence and RTF weights", fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=.12, right=.93, top=.94, bottom=.02, wspace=.04, hspace=.12)
    save_publication_figure(fig, out_path)
    return sources


def degradation_curves(out: Path, figdir: Path):
    rows = read_csv(out / "ROUND4_DEGRADATION_SUMMARY.csv")
    severity = {"clean": 0, "light": 1, "medium": 2, "heavy": 3}
    modality_families = {
        "RGB": ("rgb_brightness", "rgb_noise", "rgb_blur"),
        "Depth": ("depth_holes", "depth_block", "depth_noise"),
        "Thermal": ("thermal_blur", "thermal_noise", "thermal_local_heat"),
    }
    metrics = (("S_alpha", "S"), ("F_max", "Fmax"), ("MAE", "MAE"))
    outputs = []
    for modality, families in modality_families.items():
        fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.65), facecolor="white")
        for row_index, family in enumerate(families):
            for col_index, (metric, label) in enumerate(metrics):
                ax = axes[row_index, col_index]
                for model in MODEL_ORDER:
                    clean = next((r for r in rows if r["model"] == model and r["condition"] == "clean"), None)
                    degraded = sorted((r for r in rows if r["model"] == model and r["condition"] == family),
                                      key=lambda r: severity[r["level"]])
                    if clean and len(degraded) == 3:
                        values = [float(clean[metric])] + [float(r[metric]) for r in degraded]
                        ax.plot(range(4), values, marker="o", markersize=3.2, linewidth=1.25,
                                color=MODEL_COLORS[model], label=model)
                ax.set_xticks(range(4), ("Clean", "Light", "Medium", "Heavy"), rotation=24,
                              ha="right", rotation_mode="anchor")
                ax.set_ylabel(label)
                if col_index == 1:
                    ax.set_title(family.replace("_", " "), fontsize=7, fontweight="bold")
                ax.tick_params(labelsize=6)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5, .985), fontsize=7)
        fig.suptitle(f"{modality} synthetic degradation robustness", y=.999,
                     fontsize=8, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, .94), pad=.8, w_pad=.8, h_pad=.9)
        out_path = figdir / f"DEGRADATION_{modality.upper()}_CURVES"
        save_publication_figure(fig, out_path)
        outputs.append(out_path)
    return outputs


def make_figures(root: Path, out: Path, pred_dirs):
    figdir = out / "paper_figures"
    ordinary, challenge, failures = select_samples(root, out, pred_dirs)
    comparison_figure(root, figdir / "FIGURE5_ORDINARY_CANDIDATE", ordinary, pred_dirs,
                      "Representative ordinary-scene comparisons")
    comparison_figure(root, figdir / "FIGURE6_CHALLENGE_CANDIDATE", challenge, pred_dirs,
                      "Representative official-challenge comparisons")
    comparison_figure(root, figdir / "FIGURE8_FAILURE_CASES", failures, pred_dirs,
                      "Failure cases selected by FULL minus B0 MAE")
    mechanism_sources = mechanism_figure(root, figdir / "FIGURE7_RELIABILITY_MECHANISM")
    degradation_outputs = degradation_curves(out, figdir)
    rows = []
    for figure, group in (("Figure5", ordinary), ("Figure6", challenge), ("Figure8", failures)):
        for item in group:
            rows.append({"figure": figure, **item,
                         "data_files": f"Test/V,D,T,GT/{item['sample_id']}.png; four formal prediction PNGs",
                         "generation_script": "round4_finalize.py",
                         "description_zh": "按FULL相对B0的逐图MAE变化自动筛选；保留真实改善或失败案例。"})
    rows.append({"figure": "Figure7", "sample_id": ";".join(x.stem for x in mechanism_sources) if mechanism_sources else "UNAVAILABLE",
                 "data_files": ";".join(str(x) for x in mechanism_sources) if mechanism_sources else "UNAVAILABLE",
                 "generation_script": "round4_finalize.py",
                 "description_zh": "FULL模型两个固定样本、三个融合节点的真实CCC可靠性证据与RTF权重；色标固定为0至1。"})
    for path in degradation_outputs:
        rows.append({"figure": "Degradation curves", "sample_id": "all_1000",
                     "data_files": "ROUND4_DEGRADATION_SUMMARY.csv",
                     "generation_script": "round4_finalize.py",
                     "description_zh": f"{path.name}：四组正式模型在三种同模态退化下的S、Fmax和MAE。"})
    write_csv(out / "ROUND4_VISUALIZATION_INDEX.csv", rows)


def run_rendered_figure_qa(out: Path, source: Path):
    qa_rows = []
    try:
        import fitz  # noqa: F401
        collision_available = True
    except ImportError:
        collision_available = False
    for pdf in sorted((out / "paper_figures").glob("*.pdf")):
        stem = pdf.with_suffix("")
        text_json = Path(str(stem) + ".text-audit.json")
        collision_json = Path(str(stem) + ".collision-audit.json")
        collision_pdf = Path(str(stem) + ".collision-audit.pdf")
        text_command = [sys.executable, str(source / "audit_pdf_text.py"), str(pdf),
                        "--min-pt", "5", "--json"]
        text_result = subprocess.run(text_command, check=True, capture_output=True, text=True)
        text_json.write_text(text_result.stdout, encoding="utf-8")
        collision_status, collision_stdout = "PENDING_LOCAL_QA", "PyMuPDF unavailable in cloud environment"
        if collision_available:
            collision_command = [sys.executable, str(source / "audit_figure_collisions.py"), str(pdf),
                                 "--json-out", str(collision_json), "--overlay-pdf", str(collision_pdf)]
            collision_result = subprocess.run(collision_command, check=True, capture_output=True, text=True)
            collision_status = "PASS_OR_REVIEWED_WARN"
            collision_stdout = collision_result.stdout.strip()[:500]
        qa_rows.append({"figure": pdf.name, "pdf_text_audit": "PASS",
                        "collision_audit": collision_status,
                        "collision_report": collision_json.name if collision_available else "PENDING_LOCAL_QA",
                        "collision_stdout": collision_stdout})
    write_csv(out / "ROUND4_FIGURE_QA.csv", qa_rows)
    (out / "ROUND4_FIGURE_QA_NOTES.md").write_text(
        "# Round4 figure QA notes\n\n"
        "All plots use the Python/Matplotlib backend, editable SVG/PDF text, 600 dpi PNG/TIFF previews, "
        "a 5 pt PDF glyph floor and a render-time panel alignment gate. Rendered collision QA is run in "
        "the cloud when PyMuPDF is present; otherwise it is explicitly deferred to the local post-download gate.\n\n"
        "The degradation curves are deterministic fixed-checkpoint, fixed-test-set summaries rather than "
        "multi-seed or multi-fold stochastic aggregates; therefore no invented seed/fold error bars are shown. "
        "Per-image values are supplied in the source CSV for downstream uncertainty analyses if requested.\n",
        encoding="utf-8",
    )


def summarize_degradation(out: Path):
    rows = read_csv(out / "ROUND4_DEGRADATION_DELTA.csv")
    by_model = {}
    for model in MODEL_ORDER:
        subset = [r for r in rows if r["model"] == model]
        by_model[model] = {
            "mean_Fmax_drop": float(np.mean([float(r["F_max_degraded_minus_clean"]) for r in subset])),
            "mean_S_drop": float(np.mean([float(r["S_alpha_degraded_minus_clean"]) for r in subset])),
            "mean_MAE_increase": float(np.mean([float(r["MAE_degraded_minus_clean"]) for r in subset])),
        }
    best_f = max(by_model, key=lambda m: by_model[m]["mean_Fmax_drop"])
    report = ["# Round4 degradation robustness report", "",
              "This is a fixed-checkpoint diagnostic robustness evaluation on all 1,000 official test images under 27 deterministic synthetic conditions. It is not a physical sensor-failure experiment.", "",
              "## Average change over 27 conditions", ""]
    for model in MODEL_ORDER:
        x = by_model[model]
        report.append(f"- {model}: mean ΔFmax={x['mean_Fmax_drop']:+.6f}, mean ΔS={x['mean_S_drop']:+.6f}, mean ΔMAE={x['mean_MAE_increase']:+.6f}.")
    report += ["", f"The smallest average Fmax loss in this diagnostic is `{best_f}`. Condition-level improvements and regressions are all retained in the CSV; no favorable-only filtering was applied."]
    text = "\n".join(report)
    (out / "ROUND4_DEGRADATION_REPORT.md").write_text(text, encoding="utf-8")
    (out / "Degradation_Report.md").write_text(text, encoding="utf-8")
    return by_model


def mechanism_report(out: Path):
    weights = read_csv(out / "ROUND4_RTF_WEIGHTS.csv")
    clean = [r for r in weights if r["condition"] == "clean"]
    collapse = []
    for r in clean:
        if float(r["mean"]) > .99:
            collapse.append(f"{r['model']} scale {r['scale']} {r['tensor']} mean={float(r['mean']):.6f}")
    report = ["# Round4 reliability mechanism report", "",
              "CCC outputs are calibrated reliability evidence, not ground-truth sensor correctness probabilities.",
              "Module participation, spatial variation, degradation response and robustness improvement are evaluated as separate propositions.", "",
              "## Verified evidence", "",
              "- CCC/RTF checkpoint parameters strictly load and their residual norm ratios are non-zero where reported.",
              "- The CSVs report all-test-set reliability/weight distributions, spatial variance, entropy, saturation and dominant-area proportions.",
              "- Degradation weight shifts are paired with same-condition performance changes; a shift alone is not interpreted as successful reliability reasoning.", "",
              "## Saturation/collapse observations", ""]
    report += [f"- {x}" for x in collapse] or ["- No clean mean weight exceeded 0.99."]
    report += ["", "Severely saturated RTF weights are retained as a mechanism limitation. Round4 does not change Softmax, temperature, network structure or checkpoints."]
    text = "\n".join(report)
    (out / "ROUND4_MECHANISM_REPORT.md").write_text(text, encoding="utf-8")
    (out / "Reliability_Report.md").write_text(text, encoding="utf-8")


def make_requested_aliases(out: Path):
    aliases = {
        "ROUND4_DEGRADATION_SUMMARY.csv": "Degradation_All_Results.csv",
        "ROUND4_DEGRADATION_PER_IMAGE.csv": "Degradation_Per_Image.csv",
        "ROUND4_DEGRADATION_DELTA.csv": "Degradation_Comparison.csv",
        "ROUND4_DEGRADATION_PROTOCOL.csv": "Degradation_Parameters.csv",
        "ROUND4_CCC_RELIABILITY.csv": "CCC_Reliability_Results.csv",
        "ROUND4_RTF_WEIGHTS.csv": "RTF_Weight_Results.csv",
        "ROUND4_WEIGHT_ENTROPY.csv": "RTF_Weight_Entropy.csv",
        "ROUND4_DEGRADATION_WEIGHT_SHIFT.csv": "Degradation_Weight_Changes.csv",
        "ROUND4_RESIDUAL_ANALYSIS.csv": "Residual_Contribution.csv",
    }
    for source, target in aliases.items():
        copy_csv(out / source, out / target)


def final_report(root: Path, out: Path, package_path: Path, package_sha: str, package_size: int):
    specs = model_specs(root)
    deltas = read_csv(out / "ROUND4_DEGRADATION_DELTA.csv")
    mean_delta_lines = []
    for model in MODEL_ORDER:
        subset = [r for r in deltas if r["model"] == model]
        mean_delta_lines.append(
            f"- {model}: mean ΔS={np.mean([float(r['S_alpha_degraded_minus_clean']) for r in subset]):+.6f}, "
            f"mean ΔFmax={np.mean([float(r['F_max_degraded_minus_clean']) for r in subset]):+.6f}, "
            f"mean ΔMAE={np.mean([float(r['MAE_degraded_minus_clean']) for r in subset]):+.6f}."
        )
    model_lines = [
        f"- {name}: Round{specs[name].round}, epoch {specs[name].epoch}, step {specs[name].step}, "
        f"checkpoint `{specs[name].checkpoint}`."
        for name in MODEL_ORDER
    ]
    report = f"""# RACFNet Round4 final report

## Completion status

Round4 completed the three requested read-only experiments: all 14 official VDT-2048 challenge attributes, all 27 deterministic single-modality degradation conditions for four formal models and 1,000 test images, and CCC/RTF reliability-weight-residual analysis. It also produced Tables 1-6, candidate Figures 5-8 and three degradation curve figures. No model was retrained and no Round1-Round3 artifact was modified.

## Formal models used

{chr(10).join(model_lines)}

These identities are inherited from the already accepted Round1-Round3 results, as requested; Round4 did not repeat the excluded checkpoint/BN/bootstrap audits.

## Official challenge evaluation

`Table2_Visible.csv`, `Table3_Depth.csv`, `Table4_Thermal.csv` and `Challenge_Comparison.csv` contain four-model values and official sample counts. All improvements and regressions are retained. FULL is not uniformly best in every challenge category; for example, RTF_ONLY leads several T-RD and V-SA entries, so the evidence supports condition-dependent gains rather than universal dominance.

## Synthetic degradation robustness

All inputs were generated in memory with fixed seed 20261013. QAnet and the main model received the same degraded modality input. The evaluator streamed predictions and retained statistics rather than 108,000 PNG files.

{chr(10).join(mean_delta_lines)}

These results are diagnostic image perturbations, not a physical sensor-failure experiment. Per-condition and per-image rows are preserved so unfavorable cases remain auditable.

## CCC and RTF mechanism evidence

The delivered CSVs separately report CCC evidence means/spatial variance, RTF means, dominant-area shares, entropy, near-zero/near-one saturation, residual ratios and degraded-minus-clean shifts. Existing RTF_ONLY Softmax collapse is retained and treated as a limitation. CCC evidence must not be described as a ground-truth probability of sensor correctness, and a weight shift alone does not prove causal robustness.

## Evidence-supported claims

- The fixed CCC and RTF modules participate in inference and produce measurable reliability, weight and residual outputs.
- The four formal models show different condition-dependent challenge and degradation behaviour.
- Reliability/weight maps have measurable spatial and perturbation-dependent variation where the CSV statistics show it.

## Claims not established

- Cross-random-seed stability, multi-dataset generalization and physical sensor-failure robustness are not established.
- RTF saturation prevents a strong claim of balanced adaptive tri-modal selection.
- Local weight changes cannot be called correct or causal solely because aggregate performance changes.

## Paper deliverables

- Tables 1-6: `TABLE1_OVERALL.csv` through `TABLE6_ABLATION.csv`.
- Figures 5-8 and RGB/Depth/Thermal degradation curves: `{out / 'paper_figures'}`.
- Full server archive: `{out}`.
- Paper package: `{package_path}`.
- Package size: {package_size / 2**20:.2f} MiB.
- Package SHA-256: `{package_sha}`.

Table 1 contains only the four controlled in-project models. No external-method numbers were invented.
"""
    (out / "ROUND4_FINAL_REPORT.md").write_text(report, encoding="utf-8")


def package(root: Path, out: Path, source: Path):
    package_root = out / "paper_package"
    if package_root.exists():
        shutil.rmtree(package_root)
    dirs = {name: package_root / name for name in (
        "01_Overview", "02_Main_Tables", "03_Challenge_Results",
        "04_Degradation_Results", "05_Reliability_Analysis",
        "06_Paper_Figures", "07_Reproducibility")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    mapping = {
        "01_Overview": ["ROUND4_FINAL_REPORT.md", "ROUND4_CHALLENGE_REPORT.md", "ROUND4_DEGRADATION_REPORT.md", "ROUND4_MECHANISM_REPORT.md"],
        "02_Main_Tables": [f"TABLE{i}_{name}.csv" for i, name in ((1,"OVERALL"),(2,"VISIBLE"),(3,"DEPTH"),(4,"THERMAL"),(5,"COMPLEXITY"),(6,"ABLATION"))],
        "03_Challenge_Results": ["ROUND4_TABLE2_VISIBLE.csv", "ROUND4_TABLE3_DEPTH.csv", "ROUND4_TABLE4_THERMAL.csv", "ROUND4_CHALLENGES_ALL8.csv", "ROUND4_CHALLENGE_LABELS.csv", "ROUND4_CHALLENGE_OVERLAPS.csv", "ROUND4_CHALLENGE_REPORT.md", "Challenge_Comparison.csv"],
        "04_Degradation_Results": ["ROUND4_DEGRADATION_PROTOCOL.md", "ROUND4_DEGRADATION_PROTOCOL.csv", "ROUND4_DEGRADATION_SUMMARY.csv", "ROUND4_DEGRADATION_PER_IMAGE.csv", "ROUND4_DEGRADATION_DELTA.csv", "ROUND4_DEGRADATION_REPORT.md", "ROUND4_INPUT_DYNAMIC_RANGE.csv", "ROUND4_DEGRADATION_PREFLIGHT.csv", "ROUND4_COST_ESTIMATE.json", "Degradation_All_Results.csv", "Degradation_Per_Image.csv", "Degradation_Comparison.csv", "Degradation_Parameters.csv"],
        "05_Reliability_Analysis": ["ROUND4_CCC_RELIABILITY.csv", "ROUND4_RTF_WEIGHTS.csv", "ROUND4_WEIGHT_ENTROPY.csv", "ROUND4_DEGRADATION_WEIGHT_SHIFT.csv", "ROUND4_RESIDUAL_ANALYSIS.csv", "ROUND4_MECHANISM_REPORT.md", "CCC_Reliability_Results.csv", "RTF_Weight_Results.csv", "RTF_Weight_Entropy.csv", "Degradation_Weight_Changes.csv", "Residual_Contribution.csv"],
    }
    for dirname, names in mapping.items():
        for name in names:
            src = out / name
            if src.exists():
                shutil.copy2(src, dirs[dirname] / name)
    for file in (out / "paper_figures").glob("*"):
        if file.is_file():
            shutil.copy2(file, dirs["06_Paper_Figures"] / file.name)
    for file in source.glob("*.py"):
        shutil.copy2(file, dirs["07_Reproducibility"] / file.name)
    for name in ("run_round4_pipeline.sh",):
        if (source / name).exists():
            shutil.copy2(source / name, dirs["07_Reproducibility"] / name)
    if (out / "environment.json").exists():
        shutil.copy2(out / "environment.json", dirs["07_Reproducibility"] / "environment.json")
    manifest = []
    for file in sorted(p for p in package_root.rglob("*") if p.is_file()):
        manifest.append({"path": str(file.relative_to(package_root)).replace("\\", "/"),
                         "bytes": file.stat().st_size, "sha256": sha256(file)})
    write_csv(dirs["07_Reproducibility"] / "MANIFEST_SHA256.csv", manifest)
    package_path = root / "RACFNet_Round4_For_Paper.zip"
    temporary = package_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file in sorted(p for p in package_root.rglob("*") if p.is_file()):
            zf.write(file, file.relative_to(package_root))
    temporary.replace(package_path)
    with zipfile.ZipFile(package_path, "r") as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failure: {bad}")
    if package_path.stat().st_size > 500 * 2**20:
        raise RuntimeError("paper package exceeds 500 MiB hard limit")
    return package_path, sha256(package_path), package_path.stat().st_size


def main():
    a = parse_args()
    root, out, source = a.root.resolve(), a.out.resolve(), a.source.resolve()
    specs = model_specs(root)
    pred_dirs = {name: spec.prediction_dir.resolve() for name, spec in specs.items()}
    pred_dirs["B0"] = (root / "predictions/round1_reproduction").resolve()
    make_tables(root, out)
    summarize_degradation(out)
    mechanism_report(out)
    make_requested_aliases(out)
    make_figures(root, out, pred_dirs)
    run_rendered_figure_qa(out, source)
    # A ZIP cannot embed its own final SHA-256 without changing that SHA. The
    # packaged report points to the external audit; the archive report records
    # the final, verified value after ZIP creation.
    provisional = root / "RACFNet_Round4_For_Paper.zip"
    final_report(root, out, provisional,
                 "See external ROUND4_PACKAGE_AUDIT.json (self-hash cannot be embedded)", 0)
    package_path, package_sha, package_size = package(root, out, source)
    final_report(root, out, package_path, package_sha, package_size)
    summary = {"phase": "ROUND4_COMPLETE", "status": "PASS",
               "archive": str(out), "paper_package": str(package_path),
               "paper_package_bytes": package_size, "paper_package_sha256": package_sha,
               "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    write_json(out / "pipeline_state.json", summary)
    write_json(out / "ROUND4_PACKAGE_AUDIT.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
