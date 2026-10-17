"""Round1/Round2-compatible PySODMetrics evaluation for one Round3 mode."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
from py_sod_metrics import Emeasure, Fmeasure, MAE, Smeasure


METRICS = ("S_alpha", "MAE", "E_adp", "E_mean", "E_max", "F_adp", "F_mean", "F_max")


def measure(pairs):
    fm, em, sm, mae = Fmeasure(), Emeasure(), Smeasure(), MAE()
    for pred, gt in pairs:
        fm.step(pred=pred, gt=gt); em.step(pred=pred, gt=gt)
        sm.step(pred=pred, gt=gt); mae.step(pred=pred, gt=gt)
    f, e = fm.get_results(), em.get_results()["em"]
    fr = f["fm"]
    return {"S_alpha": float(sm.get_results()["sm"]), "MAE": float(mae.get_results()["mae"]),
            "E_adp": float(e["adp"]), "E_mean": float(np.mean(e["curve"])),
            "E_max": float(np.max(e["curve"])), "F_adp": float(fr["adp"]),
            "F_mean": float(np.mean(fr["curve"])), "F_max": float(np.max(fr["curve"]))}, f


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode", required=True, choices=("CCC_ONLY","RTF_ONLY"))
    p.add_argument("--pred-dir", type=Path, required=True); p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--baseline-metrics", type=Path, required=True)
    a=p.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    preds={x.stem:x for x in a.pred_dir.glob("*.png")}; gts={x.stem:x for x in a.gt_dir.glob("*.png")}
    if set(preds)!=set(gts) or len(preds)!=1000: raise RuntimeError("prediction/GT mismatch")
    pairs, per = [], []
    for stem in sorted(gts):
        pred, gt = cv2.imread(str(preds[stem]), 0), cv2.imread(str(gts[stem]), 0)
        if pred is None or gt is None or pred.shape != gt.shape: raise RuntimeError("bad pair " + stem)
        if pred.min() == pred.max(): raise RuntimeError("constant prediction " + stem)
        pairs.append((pred, gt)); one, _ = measure([(pred, gt)]); per.append({"image": stem, **one})
    aggregate, curves = measure(pairs)
    with a.baseline_metrics.open("r", encoding="utf-8-sig", newline="") as f: base = next(csv.DictReader(f))
    b0 = {n: float(base[n]) for n in METRICS}; delta = {n: aggregate[n]-b0[n] for n in METRICS}
    slug=a.mode.lower(); write(a.out_dir/("{}_PER_IMAGE_METRICS.csv".format(a.mode)), per)
    write(a.out_dir/("{}_EIGHT_METRICS.csv".format(a.mode)), [
        {"source": a.mode, "count": 1000, **aggregate},
        {"source": "Round1_QSFNet_B0", "count": 1000, **b0},
        {"source": a.mode+"_minus_B0", "count": 1000, **delta}])
    fr, pr = curves["fm"], curves["pr"]; curve=np.asarray(fr["curve"], float)
    precision, recall = np.asarray(pr["p"], float), np.asarray(pr["r"], float)
    write(a.out_dir/("{}_F_MEASURE_CURVE.csv".format(a.mode)),
          [{"threshold_index":i,"threshold":i/(len(curve)-1),"F_measure":float(v)} for i,v in enumerate(curve)])
    write(a.out_dir/("{}_PR_CURVE.csv".format(a.mode)),
          [{"threshold_index":i,"threshold":i/(len(curve)-1),"precision":float(precision[i]),"recall":float(recall[i])} for i in range(len(curve))])
    report={"mode":a.mode,"metrics":aggregate,"round1_b0":b0,"mode_minus_b0":delta}
    (a.out_dir/(a.mode+"_EVALUATION_REPORT.md")).write_text("# {} evaluation\n\n```json\n{}\n```\n".format(a.mode,json.dumps(report,indent=2)),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__": main()
