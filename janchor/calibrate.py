"""Build the calibration file of one model: everything J-Anchor needs besides the model itself.

1. Intervention layers: every J-lens layer L with 0.40 <= (L + 1) / n_layers <= 0.85.
2. Directions, per layer (label-free):  M_S = J_L^T (W_U[lexicon_S] - mean_vocab)^T,  S in {R, H}.
   Refusal: the left singular vectors covering 90% of the energy (at most --kmax); harmfulness: the leading one.
   Every axis is signed so that it points toward the mean pulled-back lexicon word.
3. Benign statistics: data/wjb_benign_calib500.jsonl (WildJailbreak vanilla_benign, seed 0), each prompt followed by
   its reference response (<= 512 tokens), teacher-forced in one forward pass.  For every axis: mean / std of the
   cosine at the assistant role token (anchor) and pooled over all response positions.
4. Gate layers: the three layers whose anchor score max(z_R1, z_H) best separates data/advbench_dev420.jsonl from
   data/wjb_benign_dev100.jsonl (AUROC).
-> calib/<model>.pt
run: python -m janchor.calibrate --model llama
"""
from __future__ import annotations
import argparse, json, os, time
import torch
from . import models as M
from .lexicon import refusal_ids, harm_ids

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def read_jsonl(path):
    return [json.loads(l) for l in open(path)]


def signed_svd(J, UC, k):
    """Top-k left singular vectors of J^T UC^T as rows, each signed toward J^T mean(UC); and the cumulative energy."""
    U, s, _ = torch.linalg.svd(J.T @ UC.T, full_matrices=False)
    um = UC.mean(0) @ J
    U = U * torch.sign((U * um[:, None]).sum(0, keepdim=True))
    e = s ** 2
    return U[:, :k].T.contiguous(), (e / e.sum()).cumsum(0)[:k]


def left_pad(seqs, pad_id):
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long); am = torch.zeros(len(seqs), T, dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, T - len(s):] = torch.tensor(s); am[j, T - len(s):] = 1
    return ids.cuda(), am.cuda()


def auroc(pos, neg):
    s = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg]); n1, n0 = len(pos), len(neg)
    return (sum(i + 1 for i, (_, y) in enumerate(s) if y) - n1 * (n1 + 1) / 2) / (n1 * n0)


class _Moments:
    def __init__(self, k): self.s = torch.zeros(k, dtype=torch.float64, device="cuda"); self.ss = self.s.clone(); self.n = 0
    def add(self, v):                     # v: [N, k]
        v = v.double(); self.s += v.sum(0); self.ss += (v ** 2).sum(0); self.n += v.shape[0]
    def get(self):
        m = self.s / self.n
        return m.float().cpu(), (self.ss / self.n - m ** 2).clamp(min=1e-12).sqrt().float().cpu()


@torch.no_grad()
def calibrate(key, out_path, energy=0.90, kmax=24, depth=(0.40, 0.85), batch=8, max_resp=512, n_gate=3, lens=None):
    spec = M.get_spec(key, lens); t0 = time.time()
    tok, hf = M.load(spec)
    blocks = M.blocks(hf); n_layers = len(blocks)
    J = torch.load(spec.lens, map_location="cpu", weights_only=True)["J"]
    layers = [L for L in sorted(J) if depth[0] <= (L + 1) / n_layers <= depth[1]]
    WU = M.unembedding(hf).float(); mu = WU.mean(0)
    ids = {"R": refusal_ids(tok, WU.shape[0]), "H": harm_ids(tok, WU.shape[0])}
    UC = {k: (WU[torch.tensor(v, device=WU.device)] - mu) for k, v in ids.items()}
    UR, uH, rank, cum_R = {}, {}, {}, {}
    for L in layers:
        JL = J[L].float().cuda()
        basis, cum = signed_svd(JL, UC["R"], kmax)
        r = min(int((cum < energy).sum()) + 1, basis.shape[0])
        UR[L], rank[L], cum_R[L] = basis[:r], r, cum.cpu()
        uH[L] = signed_svd(JL, UC["H"], 1)[0][0]
        del JL
    del J, WU; torch.cuda.empty_cache()
    roff = M.role_offset(tok, spec)
    print(f"[{key}] {n_layers} blocks, layers {layers[0]}-{layers[-1]} ({len(layers)}), |R|={len(ids['R'])} |H|={len(ids['H'])}, "
          f"rank {min(rank.values())}-{max(rank.values())}, role offset {roff}", flush=True)

    cap = {}
    hooks = [blocks[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, o if torch.is_tensor(o) else o[0])) for L in layers]
    proj = lambda L, h: torch.cat([h @ UR[L].T, (h @ uH[L])[..., None]], -1)       # [..., r+1]: refusal axes, then H

    # benign statistics (teacher-forced reference responses)
    rows = read_jsonl(os.path.join(DATA, "wjb_benign_calib500.jsonl"))
    anc = {L: _Moments(rank[L] + 1) for L in layers}; resp = {L: _Moments(rank[L] + 1) for L in layers}
    for i in range(0, len(rows), batch):
        ch = rows[i:i + batch]
        pr = [M.encode_prompt(tok, spec, r["prompt"]) for r in ch]
        rs = [tok.encode(r["response"], add_special_tokens=False)[:max_resp] for r in ch]
        ids_, am = left_pad([p + q for p, q in zip(pr, rs)], tok.pad_token_id); T = ids_.shape[1]
        nr = torch.tensor([len(q) for q in rs], device="cuda")
        is_resp = torch.arange(T, device="cuda")[None, :] >= (T - nr[:, None])
        role = T - 1 - nr - roff
        hf(input_ids=ids_, attention_mask=am, position_ids=(am.cumsum(1) - 1).clamp(min=0), logits_to_keep=1)
        for L in layers:
            h = cap[L].float(); c = proj(L, h / h.norm(dim=-1, keepdim=True))
            resp[L].add(c[is_resp]); anc[L].add(c[torch.arange(len(ch), device="cuda"), role])
    null = {}
    for L in layers:
        ma, sa = anc[L].get(); mg, sg = resp[L].get(); r = rank[L]
        null[L] = {"R": {"mu_a": ma[:r], "sd_a": sa[:r], "mu_g": mg[:r], "sd_g": sg[:r]},
                   "H": {"mu_a": ma[r], "sd_a": sa[r], "mu_g": mg[r], "sd_g": sg[r]}}
    print(f"[{key}] benign statistics from {len(rows)} prompts ({time.time() - t0:.0f}s)", flush=True)

    # gate layers: anchor score max(z_R1, z_H) on development prompts
    def anchor_scores(prompts):
        out = {L: [] for L in layers}
        for i in range(0, len(prompts), 2 * batch):
            seqs = [M.encode_prompt(tok, spec, p) for p in prompts[i:i + 2 * batch]]
            ids_, am = left_pad(seqs, tok.pad_token_id); T = ids_.shape[1]
            hf(input_ids=ids_, attention_mask=am, position_ids=(am.cumsum(1) - 1).clamp(min=0), logits_to_keep=1)
            for L in layers:
                h = cap[L][:, T - 1 - roff].float(); h = h / h.norm(dim=-1, keepdim=True)
                zr = (h @ UR[L][0] - null[L]["R"]["mu_a"][0].cuda()) / null[L]["R"]["sd_a"][0].cuda()
                zh = (h @ uH[L] - null[L]["H"]["mu_a"].cuda()) / null[L]["H"]["sd_a"].cuda()
                out[L] += torch.maximum(zr, zh).tolist()
        return out
    pos = anchor_scores([r["prompt"] for r in read_jsonl(os.path.join(DATA, "advbench_dev420.jsonl"))])
    neg = anchor_scores([r["prompt"] for r in read_jsonl(os.path.join(DATA, "wjb_benign_dev100.jsonl"))])
    au = {L: auroc(pos[L], neg[L]) for L in layers}
    gate_layers = sorted(sorted(layers, key=lambda L: -au[L])[:n_gate])
    for h in hooks: h.remove()
    print(f"[{key}] gate layers {gate_layers} (AUROC " + ", ".join(f"L{L} {au[L]:.3f}" for L in gate_layers) + ")", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({"meta": {"model": key, "repo": spec.repo, "lens": spec.lens, "n_layers": n_layers, "role_offset": roff,
                         "energy": energy, "kmax": kmax, "depth": list(depth), "n_calib": len(rows), "max_resp": max_resp,
                         "n_refusal_tokens": len(ids["R"]), "n_harm_tokens": len(ids["H"])},
                "layers": layers, "rank": rank, "cum_R": cum_R,
                "UR": {L: UR[L].cpu() for L in layers}, "uH": {L: uH[L].cpu() for L in layers}, "null": null,
                "gate": {"layers": gate_layers, "auroc": au}}, out_path)
    print(f"[{key}] wrote {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=f"one of {', '.join(M.MODELS)}, or an HF repo id (with --lens)")
    ap.add_argument("--lens", default=None, help="J-lens .pt file (required for a model that is not registered)")
    ap.add_argument("--out", default="", help="default: calib/<model>.pt")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    spec = M.get_spec(a.model, a.lens)
    calibrate(a.model, a.out or os.path.join(ROOT, "calib", f"{spec.key}.pt"), batch=a.batch, lens=a.lens)
