from __future__ import annotations
import argparse, csv, hashlib, json, shutil
from pathlib import Path
import numpy as np

METRICS=("S_alpha","MAE","E_adp","E_mean","E_max","F_adp","F_mean","F_max")
def read_rows(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def metrics(path,source):
    rows=read_rows(path); row=next(r for r in rows if r["source"]==source)
    return {m:float(row[m]) for m in METRICS}
def first_metrics(path):
    row=read_rows(path)[0];return {m:float(row[m]) for m in METRICS}
def per(path):
    rows=read_rows(path);return {r["image"]:{m:float(r[m]) for m in METRICS} for r in rows}

def main():
    p=argparse.ArgumentParser();p.add_argument("--work-root",type=Path,required=True);p.add_argument("--bootstrap",type=int,default=2000);p.add_argument("--seed",type=int,default=20261013);a=p.parse_args()
    root=a.work_root.resolve(); r3=root/"round3"; out=r3/"paper_results"; out.mkdir(parents=True,exist_ok=True)
    b0=first_metrics(root/"paper_results"/"round1"/"ROUND1_BASELINE_METRICS.csv")
    full=metrics(root/"round2"/"paper_results"/"ROUND2_FULL_EIGHT_METRICS.csv","RACFNet_FULL")
    ccc=metrics(out/"ccc_only"/"CCC_ONLY_EIGHT_METRICS.csv","CCC_ONLY")
    rtf=metrics(out/"rtf_only"/"RTF_ONLY_EIGHT_METRICS.csv","RTF_ONLY")
    ckpts={"CCC_ONLY":r3/"checkpoints"/"ccc_only"/"RACFNet_CCC_ONLY_final.pth",
           "RTF_ONLY":r3/"checkpoints"/"rtf_only"/"RACFNet_RTF_ONLY_final.pth",
           "FULL":root/"round2"/"checkpoints"/"RACFNet_FULL_final.pth",
           "B0":root/"checkpoints"/"Tnet_final.pth"}
    table=[]
    for mode,m in (("B0",b0),("CCC_ONLY",ccc),("RTF_ONLY",rtf),("FULL",full)):
        table.append({"model":mode,"epochs":200,"steps":52400,"seed":20261013,"checkpoint":str(ckpts[mode]),"checkpoint_sha256":sha(ckpts[mode]),**m})
    write(out/"ROUND3_EIGHT_METRICS.csv",table);write(out/"ROUND3_ABLATION_TABLE.csv",table)
    deltas=[]
    for label,left,right in (("CCC_ONLY_minus_B0",ccc,b0),("RTF_ONLY_minus_B0",rtf,b0),("FULL_minus_CCC_ONLY",full,ccc),("FULL_minus_RTF_ONLY",full,rtf)):
        deltas.append({"comparison":label,**{m:left[m]-right[m] for m in METRICS}})
    write(out/"ROUND3_DELTA_ANALYSIS.csv",deltas)

    sources={"B0":per(root/"paper_results"/"round1"/"ROUND1_PER_IMAGE_METRICS.csv"),
             "FULL":per(root/"round2"/"paper_results"/"ROUND2_FULL_PER_IMAGE_METRICS.csv"),
             "CCC_ONLY":per(out/"ccc_only"/"CCC_ONLY_PER_IMAGE_METRICS.csv"),
             "RTF_ONLY":per(out/"rtf_only"/"RTF_ONLY_PER_IMAGE_METRICS.csv")}
    names=set(sources["B0"])
    if len(names)!=1000 or any(set(v)!=names for v in sources.values()):raise RuntimeError("per-image stem mismatch")
    combined=[]
    for image in sorted(names):
        for model in ("B0","CCC_ONLY","RTF_ONLY","FULL"):
            combined.append({"image":image,"model":model,**sources[model][image]})
    write(out/"ROUND3_PER_IMAGE_METRICS.csv",combined)
    rng=np.random.default_rng(a.seed); order=sorted(names); n=len(order); boot=[]
    comparisons=(("CCC_ONLY","B0"),("RTF_ONLY","B0"),("FULL","CCC_ONLY"),("FULL","RTF_ONLY"),("FULL","B0"))
    for left,right in comparisons:
        for metric in METRICS:
            diff=np.asarray([sources[left][x][metric]-sources[right][x][metric] for x in order],dtype=float)
            samples=np.empty(a.bootstrap,dtype=float)
            for i in range(a.bootstrap):samples[i]=diff[rng.integers(0,n,n)].mean()
            boot.append({"left":left,"right":right,"metric":metric,"paired_mean_difference":float(diff.mean()),
                         "ci95_low":float(np.quantile(samples,.025)),"ci95_high":float(np.quantile(samples,.975)),
                         "bootstrap_samples":a.bootstrap,"seed":a.seed,
                         "scope":"fixed trained models; test-image sampling uncertainty only"})
    write(out/"ROUND3_PAIRED_BOOTSTRAP.csv",boot)

    ccc_mech=read_rows(r3/"reliability_artifacts"/"ccc_only"/"mechanism_statistics.csv")
    rtf_mech=read_rows(r3/"reliability_artifacts"/"rtf_only"/"mechanism_statistics.csv")
    write(out/"ROUND3_CCC_RELIABILITY_STATISTICS.csv",ccc_mech)
    write(out/"ROUND3_RTF_WEIGHT_STATISTICS.csv",rtf_mech)
    residual=read_rows(r3/"reliability_artifacts"/"ccc_only"/"residual_norm_statistics.csv")+read_rows(r3/"reliability_artifacts"/"rtf_only"/"residual_norm_statistics.csv")
    write(out/"ROUND3_RESIDUAL_NORM_STATISTICS.csv",residual)
    visuals=read_rows(r3/"reliability_artifacts"/"ccc_only"/"visualization_index.csv")+read_rows(r3/"reliability_artifacts"/"rtf_only"/"visualization_index.csv")
    write(out/"ROUND3_VISUALIZATION_INDEX.csv",visuals)
    rtf_weights=[r for r in rtf_mech if r["tensor"] in ("Wv","Wd","Wt")]
    summary={}
    for scale in range(3):
        summary[str(scale)]={}
        for token in ("Wv","Wd","Wt"):
            rows=[r for r in rtf_weights if int(r["scale"])==scale and r["tensor"]==token]
            summary[str(scale)][token]={"mean":float(np.mean([float(r["mean"]) for r in rows])),
                                        "dominant_area_ratio":float(np.mean([float(r["dominant_area_ratio"]) for r in rows])),
                                        "saturated_high_ratio":float(np.mean([float(r["saturated_high_ratio"]) for r in rows]))}
    (out/"ROUND3_WEIGHT_COLLAPSE_ANALYSIS.md").write_text("# RTF_ONLY weight-collapse analysis\n\n```json\n"+json.dumps(summary,indent=2)+"\n```\n\nA near-one RGB weight and near-zero Depth/Thermal weights are reported as a mechanism limitation, not hidden.\n",encoding="utf-8")
    report={"status":"COMPLETE","metrics":{"B0":b0,"CCC_ONLY":ccc,"RTF_ONLY":rtf,"FULL":full},"checkpoint_hashes":{k:sha(v) for k,v in ckpts.items()},"metric_sources":{"B0":str(root/"paper_results"/"round1"/"ROUND1_BASELINE_METRICS.csv"),"FULL":str(root/"round2"/"paper_results"/"ROUND2_FULL_EIGHT_METRICS.csv")},"bootstrap_note":"95% CIs quantify test-image sampling uncertainty for fixed single-seed trained models; they do not establish cross-seed stability."}
    (out/"ROUND3_REPORT.md").write_text("# RACFNet Round3 report\n\n```json\n"+json.dumps(report,indent=2)+"\n```\n",encoding="utf-8")
    audit=["# Round3 checkpoint audit","","All formal ablations initialize from the same Round1 Tnet checkpoint. Smoke weights are not used for formal training.",""]
    for key,path in ckpts.items():audit.append("- `{}`: `{}`  SHA-256 `{}`".format(key,path,sha(path)))
    (out/"ROUND3_CHECKPOINT_AUDIT.md").write_text("\n".join(audit)+"\n",encoding="utf-8")
    configs=out/"ROUND3_TRAINING_CONFIGS";configs.mkdir(exist_ok=True)
    for mode in ("ccc_only","rtf_only"):
        shutil.copy2(r3/"experiments"/mode/"config.json",configs/(mode+"_config.json"))
    shutil.copy2(root/"round2"/"source"/"RACFNet"/"round3"/"ROUND3_METHOD_CONSISTENCY_REPORT.md",out/"ROUND3_METHOD_CONSISTENCY_REPORT.md")
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":main()
