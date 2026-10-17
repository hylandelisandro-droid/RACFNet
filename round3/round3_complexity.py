from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--work-root",type=Path,required=True);p.add_argument("--qanet",type=Path,required=True);a=p.parse_args()
    repo=a.repo.resolve();sys.path[:0]=[str(repo),str(repo/"round2")]
    from QAnet import QAnet
    from round2.RACFNet import RACFNet
    from round2.round2_common import load_strict,write_csv
    root=a.work_root.resolve();qnet=QAnet().cuda().eval();load_strict(qnet,a.qanet,"QAnet complexity")
    inputs=tuple(torch.randn(1,3,384,384,device="cuda") for _ in range(3));rows=[]
    for mode in ("RTF_ONLY","CCC_ONLY"):
        model=RACFNet(mode).cuda().eval(); ckpt=root/"round3"/"checkpoints"/mode.lower()/("RACFNet_{}_final.pth".format(mode));load_strict(model,ckpt,mode)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(10):
                dl,tl=qnet(*inputs);model(*inputs,dl,tl)
            torch.cuda.synchronize();tic=time.perf_counter()
            for _ in range(50):
                dl,tl=qnet(*inputs);model(*inputs,dl,tl)
            torch.cuda.synchronize();lat=(time.perf_counter()-tic)/50
        flops="UNAVAILABLE";flops_note="thop unavailable or unsupported operator"
        try:
            from thop import profile
            with torch.no_grad():dl,tl=qnet(*inputs); macs,_=profile(model,inputs=(*inputs,dl,tl),verbose=False)
            flops=float(2*macs);flops_note="2 x MACs from thop; QAnet excluded from FLOPs"
        except Exception as exc: flops_note="{}: {}".format(type(exc).__name__,str(exc)[:200])
        prefixes=("ccc_scales.",) if mode=="CCC_ONLY" else ("rtf_scales.","rtf_quality_v.")
        rows.append({"mode":mode,"input":"1x3x384x384 RGB,D,T","parameters":sum(x.numel() for x in model.parameters()),
                     "new_parameters":sum(x.numel() for n,x in model.named_parameters() if n.startswith(prefixes)),
                     "FLOPs":flops,"FLOPs_note":flops_note,"mean_latency_ms":lat*1000,"FPS":1/lat,
                     "peak_vram_mib":torch.cuda.max_memory_allocated()/2**20,"hardware":torch.cuda.get_device_name(0),
                     "timing":"batch1 FP32, 10 warmup, 50 timed; QAnet + model"})
        del model;torch.cuda.empty_cache()
    write_csv(root/"round3"/"paper_results"/"ROUND3_COMPLEXITY.csv",rows);print(json.dumps(rows,indent=2),flush=True)
if __name__=="__main__":main()
