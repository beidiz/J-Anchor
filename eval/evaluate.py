"""Generate and judge the evaluation suites of one model; report every dataset separately (no averages).

    suite    datasets ($JANCHOR_EVAL_DATA/<suite>/)       prefill                    max_new   metric
    fixed    strongreject, advbench, harmbench           fixed template             512       ASR   (HarmBench-Llama-2-13b-cls)
    deep     strongreject, advbench, harmbench           first 500 tokens of row    512       ASR
    benign   xstest, oktest                              none                       256       ORR   (WildGuard refusal)
    math     gsm8k, math500                              none                       1024/2048 accuracy (math-verify)

Everything is written under --out: <suite>/<dataset>.jsonl (generations), <suite>/<dataset>.judged.jsonl (with the
label), summary.json and summary.md.  Each stage resumes: finished generations and judged files are reused.
examples:
    python -m eval.evaluate --model llama --out runs/llama
    python -m eval.evaluate --model llama --suites fixed,benign --no_defense --out runs/llama_undefended
"""
from __future__ import annotations
import argparse, gc, json, os, runpy, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = runpy.run_path(os.path.join(ROOT, "config.py"))["EVAL_DATA"]
SUITES = {
    "fixed": {"datasets": ["strongreject", "advbench", "harmbench"], "prefill": "fixed", "prefill_tokens": 0, "max_new": {}, "metric": "ASR"},
    "deep": {"datasets": ["strongreject", "advbench", "harmbench"], "prefill": "field", "prefill_tokens": 500, "max_new": {}, "metric": "ASR"},
    "benign": {"datasets": ["xstest", "oktest"], "prefill": "none", "prefill_tokens": 0, "max_new": {"xstest": 256, "oktest": 256}, "metric": "ORR"},
    "math": {"datasets": ["gsm8k", "math500"], "prefill": "none", "prefill_tokens": 0, "max_new": {"gsm8k": 1024, "math500": 2048}, "metric": "accuracy"},
}
LABEL = {"ASR": "harmful", "ORR": "refusal", "accuracy": "correct"}
read = lambda p: [json.loads(l) for l in open(p)]


def write_judged(path, rows, key, labels):
    with open(path, "w") as f:
        for r, y in zip(rows, labels):
            f.write(json.dumps(r | {key: y}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a registered key, or an HF repo id (with --lens)")
    ap.add_argument("--lens", default=None, help="J-lens .pt file (required for a model that is not registered)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default="fixed,deep,benign,math")
    ap.add_argument("--rho", type=float, default=None); ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--gate", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--no_defense", action="store_true"); ap.add_argument("--calib", default="")
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--no_repeat", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="only the first N rows of every dataset (smoke test)")
    a = ap.parse_args()
    suites = [s.strip() for s in a.suites.split(",") if s.strip()]
    for s in suites:
        if s not in SUITES: raise SystemExit(f"unknown suite {s!r}; choose from {', '.join(SUITES)}")
    jobs = [(s, d) for s in suites for d in SUITES[s]["datasets"]]
    gen = lambda s, d: os.path.join(a.out, s, f"{d}.jsonl")
    judged = lambda s, d: os.path.join(a.out, s, f"{d}.judged.jsonl")
    n_rows = lambda s, d: len(read(os.path.join(EVAL, s, f"{d}.jsonl"))[:a.n or None])
    complete = lambda p, n: os.path.exists(p) and sum(1 for _ in open(p)) >= n

    # 1. generation (the model is loaded once)
    todo = [(s, d) for s, d in jobs if not complete(gen(s, d), n_rows(s, d))]
    if todo:
        import torch
        from janchor.generate import load_defended, generate_file
        spec, tok, hf, ja = load_defended(a.model, a.rho, a.tau, a.gate, a.no_defense, a.calib, a.lens)
        for s, d in todo:
            cfg = SUITES[s]; os.makedirs(os.path.join(a.out, s), exist_ok=True)
            generate_file(spec, tok, hf, ja, os.path.join(EVAL, s, f"{d}.jsonl"), gen(s, d), cfg["prefill"], cfg["prefill_tokens"],
                          cfg["max_new"].get(d, 512), a.batch, a.no_repeat, a.n)
        del spec, tok, hf, ja; gc.collect(); torch.cuda.empty_cache()

    # 2. judging (one judge at a time, nothing else on the GPU)
    by_metric = lambda m: [(s, d) for s, d in jobs if SUITES[s]["metric"] == m and not complete(judged(s, d), n_rows(s, d))]
    asr = by_metric("ASR")
    if asr:
        from .judge_asr import judge_asr
        rows = {j: read(gen(*j)) for j in asr}
        labels = judge_asr([r for j in asr for r in rows[j]])
        k = 0
        for j in asr:
            write_judged(judged(*j), rows[j], "harmful", labels[k:k + len(rows[j])]); k += len(rows[j])
    orr = by_metric("ORR")
    if orr:
        from .judge_orr import judge_orr
        rows = {j: read(gen(*j)) for j in orr}
        labels = judge_orr([r for j in orr for r in rows[j]], os.path.join(a.out, "wildguard"), "benign_" + "_".join(d for _, d in orr))
        k = 0
        for j in orr:
            write_judged(judged(*j), rows[j], "refusal", labels[k:k + len(rows[j])]); k += len(rows[j])
    mth = by_metric("accuracy")
    if mth:
        files = [os.path.abspath(gen(*j)) for j in mth]
        res = subprocess.run(["uv", "run", "--no-project", "--python", "3.12.14", "--with", "math-verify==0.7.0", "python",
                              os.path.join(ROOT, "eval", "judge_math.py"), *files], capture_output=True, text=True, check=True)
        got = json.loads(res.stdout.strip().splitlines()[-1])
        for j, f in zip(mth, files):
            write_judged(judged(*j), read(gen(*j)), "correct", got[f])

    # 3. summary, one row per dataset
    summary = []
    for s, d in jobs:
        rows = read(judged(s, d)); m = SUITES[s]["metric"]; key = LABEL[m]
        e = {"suite": s, "dataset": d, "metric": m, "value": round(100 * sum(r[key] for r in rows) / len(rows), 1), "n": len(rows)}
        if rows and "protected" in rows[0]:
            e["protected"] = round(100 * sum(r["protected"] for r in rows) / len(rows), 1)
        summary.append(e)
    from janchor.models import get_spec
    spec = get_spec(a.model, a.lens)
    defense = "none" if a.no_defense else {"rho": a.rho if a.rho is not None else spec.rho, "tau": a.tau,
                                           "gate": spec.gate if a.gate == "auto" else a.gate == "on"}
    json.dump({"model": a.model, "defense": defense,
               "results": summary}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    lines = [f"# {a.model}: " + ("undefended" if a.no_defense else f"J-Anchor rho={defense['rho']:g}, tau={a.tau:g}, gate {'on' if defense['gate'] else 'off'}"), "",
             "| suite | dataset | metric | % | n | gate fired % |", "|---|---|---|---|---|---|"]
    lines += [f"| {e['suite']} | {e['dataset']} | {e['metric']} | {e['value']} | {e['n']} | {e.get('protected', '-')} |" for e in summary]
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
