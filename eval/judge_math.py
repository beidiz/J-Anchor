"""Reasoning accuracy with math-verify: the gold answer and the full response are parsed and compared symbolically.
evaluate.py runs it as
    uv run --no-project --python 3.12.14 --with math-verify==0.7.0 python eval/judge_math.py FILE...
Prints {file: [0/1 per row]} as json.
"""
import json, logging, os, sys

logging.disable(logging.CRITICAL)
from math_verify import parse, verify


def correct(response: str, gold: str, dataset: str) -> int:
    try:
        return int(bool(verify(parse(f"${gold}$" if dataset == "math500" else gold), parse(response))))
    except Exception:
        return 0


if __name__ == "__main__":
    out = {}
    for path in sys.argv[1:]:
        rows = [json.loads(l) for l in open(path)]
        ds = "gsm8k" if "gsm8k" in os.path.basename(path) else "math500"
        out[path] = [correct(r["response"], r["answer"], ds) for r in rows]
    print(json.dumps(out))
