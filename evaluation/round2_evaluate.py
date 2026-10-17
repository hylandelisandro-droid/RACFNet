"""Evaluate a complete VDT-2048 prediction set and emit aggregate, per-image and curves."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np
from py_sod_metrics import Emeasure,Fmeasure,MAE,Smeasure

def metrics_for(pairs):
    fm,em,sm,mae=Fmeasure(),Emeasure(),Smeasure(),MAE()
    for pred,gt in pairs: fm.step(pred=pred,gt=gt); em.step(pred=pred,gt=gt); sm.step(pred=pred,gt=gt); mae.step(pred=pred,gt=gt)
    f_all=fm.get_results(); fr=f_all["fm"]; er=em.get_results()["em"]
    return {"S_alpha":float(sm.get_results()["sm"]),"MAE":float(mae.get_results()["mae"]),"E_adp":float(er["adp"]),"E_mean":float(np.mean(er["curve"])),"E_max":float(np.max(er["curve"])),"F_adp":float(fr["adp"]),"F_mean":float(np.mean(fr["curve"])),"F_max":float(np.max(fr["curve"]))},f_all,er

def write(path,rows):
    with path.open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pred-dir",type=Path,required=True); ap.add_argument("--gt-dir",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); ap.add_argument("--source",required=True); ap.add_argument("--baseline-metrics",type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    preds={x.stem:x for x in a.pred_dir.glob("*.png")}; gts={x.stem:x for x in a.gt_dir.glob("*.png")}
    if set(preds)!=set(gts) or len(preds)!=1000: raise SystemExit(f"set mismatch: pred={len(preds)} gt={len(gts)} missing={len(set(gts)-set(preds))} extra={len(set(preds)-set(gts))}")
    arrays=[]; per=[]
    for stem in sorted(gts):
        p=cv2.imread(str(preds[stem]),0); g=cv2.imread(str(gts[stem]),0)
        if p is None or g is None or p.shape!=g.shape: raise RuntimeError(f"bad pair {stem}")
        arrays.append((p,g)); m,_,_=metrics_for([(p,g)]); per.append({"image":stem,**m})
    m,f_all,er=metrics_for(arrays); fr=f_all["fm"]; pr=f_all["pr"]; write(a.out_dir/"ROUND2_FULL_PER_IMAGE_METRICS.csv",per)
    with a.baseline_metrics.open("r",encoding="utf-8-sig",newline="") as f: baseline=next(csv.DictReader(f))
    metric_names=("S_alpha","MAE","E_adp","E_mean","E_max","F_adp","F_mean","F_max")
    baseline_row={name:float(baseline[name]) for name in metric_names}
    delta={name:m[name]-baseline_row[name] for name in metric_names}
    write(a.out_dir/"ROUND2_FULL_EIGHT_METRICS.csv",[
        {"source":a.source,"count":1000,**m},
        {"source":"Round1_QSFNet_B0","count":1000,**baseline_row},
        {"source":"FULL_minus_B0","count":1000,**delta},
    ])
    curve=np.asarray(fr["curve"],dtype=float); precision=np.asarray(pr["p"],dtype=float); recall=np.asarray(pr["r"],dtype=float)
    if precision.size!=curve.size: precision=np.full(curve.size,np.nan)
    if recall.size!=curve.size: recall=np.full(curve.size,np.nan)
    write(a.out_dir/"ROUND2_F_MEASURE_CURVE.csv",[{"threshold_index":i,"threshold":i/(len(curve)-1),"F_measure":float(v)} for i,v in enumerate(curve)])
    write(a.out_dir/"ROUND2_PR_CURVE.csv",[{"threshold_index":i,"threshold":i/(len(curve)-1),"precision":float(precision[i]),"recall":float(recall[i])} for i in range(len(curve))])
    report={"full":m,"round1_b0":baseline_row,"full_minus_b0":delta}
    (a.out_dir/"ROUND2_EVALUATION_REPORT.md").write_text("# RACFNet Round2 FULL evaluation\n\n```json\n"+json.dumps(report,indent=2)+"\n```\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
