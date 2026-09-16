#!/usr/bin/env python3
"""Reproducible CPU/CUDA and quantization benchmark with CSV/JSON output."""
import argparse, csv, hashlib, json, os, re, statistics, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def run_case(binary, backend, model, tokenizer, prompt, predict, threads, quant, device):
    cmd=[str(binary),"generate","--model",str(model),"--tokenizer",str(tokenizer),
         "--backend",backend,"--threads",str(threads),
         "--temperature","0","--seed","1","--prompt",prompt,"--n-predict",str(predict)]
    if backend == "cuda": cmd += ["--device",str(device)]
    if quant: cmd += ["--quant",quant]
    mem_before=mem_after=None
    if backend == "cuda":
        try: mem_before=int(subprocess.check_output(["nvidia-smi",f"--id={device}","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
        except (OSError,subprocess.SubprocessError,ValueError): pass
    start=time.perf_counter(); p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True); elapsed=time.perf_counter()-start
    if backend == "cuda":
        try: mem_after=int(subprocess.check_output(["nvidia-smi",f"--id={device}","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
        except (OSError,subprocess.SubprocessError,ValueError): pass
    text=p.stdout+"\n"+p.stderr
    if p.returncode: raise RuntimeError(f"failed: {' '.join(cmd)}\n{text}")
    def metric(name):
        m=re.search(rf"stage={name} .*?use_time_ms=([0-9.]+)",text); return float(m.group(1))/1000 if m else None
    pre=metric("prefill_ms"); dec=metric("decode_ms")
    return {"elapsed_s":elapsed,"gpu_mem_before_mib":mem_before,"gpu_mem_after_mib":mem_after,"prefill_s":pre,"decode_s":dec,"prefill_tok_s":(6/pre if pre else None),
            "decode_tok_s":(predict/dec if dec else None),"returncode":p.returncode,"command":" ".join(cmd)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--binary",default=str(ROOT/"build-cpu"/"mini-llama")); ap.add_argument("--cuda-binary"); ap.add_argument("--model",type=Path,default=ROOT/"models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf"); ap.add_argument("--tokenizer",type=Path,default=ROOT/"models/chat/vocab.json"); ap.add_argument("--out",type=Path,default=ROOT/"result"/"benchmark"); ap.add_argument("--repeats",type=int,default=5); ap.add_argument("--warmup",type=int,default=2); ap.add_argument("--predict",type=int,default=16); ap.add_argument("--device",type=int,default=0); ap.add_argument("--threads",default="1,4"); ap.add_argument("--quant",default="f32,q8_0,q4_0", help="comma-separated formats supported by CLI: f32,q8_0,q4_0"); args=ap.parse_args()
    requested_quant = [value.strip().lower() for value in args.quant.split(",") if value.strip()]
    unsupported = sorted(set(requested_quant) - {"f32", "q8_0", "q4_0"})
    if unsupported:
        ap.error("unsupported quantization format(s): " + ", ".join(unsupported) +
                 ". The generate CLI currently supports only f32, q8_0, q4_0.")
    args.out.mkdir(parents=True,exist_ok=True); run_id=time.strftime("%Y%m%d_%H%M%S"); rows=[]
    bins=[("cpu",Path(args.binary),t) for t in map(int,args.threads.split(","))]
    if args.cuda_binary: bins.append(("cuda",Path(args.cuda_binary),1))
    for backend,binary,threads in bins:
      for quant in requested_quant:
       q=None if quant in ("f32","model-native") else quant
       for _ in range(args.warmup): run_case(binary,backend,args.model,args.tokenizer,"hello",args.predict,threads,q,args.device)
       for i in range(args.repeats):
        r=run_case(binary,backend,args.model,args.tokenizer,"hello",args.predict,threads,q,args.device); r.update({"run_id":run_id,"backend":backend,"quant":quant,"threads":threads,"repeat":i,"model":str(args.model)}); rows.append(r)
    outcsv=args.out/f"{run_id}_raw_timings.csv"
    with outcsv.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    groups={}
    for r in rows: groups.setdefault((r["backend"],r["quant"],r["threads"]),[]).append(r)
    summary=[]
    for key,vals in groups.items():
      summary.append({"backend":key[0],"quant":key[1],"threads":key[2],"samples":len(vals),"elapsed_median_s":statistics.median(v["elapsed_s"] for v in vals),"prefill_tok_s_median":statistics.median(v["prefill_tok_s"] for v in vals if v["prefill_tok_s"]),"decode_tok_s_median":statistics.median(v["decode_tok_s"] for v in vals if v["decode_tok_s"])})
    (args.out/f"{run_id}_summary.json").write_text(json.dumps({"run_id":run_id,"model":str(args.model),"model_sha256":sha(Path(args.model)),"samples":summary},indent=2),encoding="utf-8")
    print(f"PASS benchmark run_id={run_id} raw={outcsv}")
if __name__=="__main__": main()

