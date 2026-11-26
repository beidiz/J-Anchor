"""J-Anchor: request-gated, anchor-targeted refusal restoration inside the J-lens refusal subspace.

For every intervention layer l and refusal axis i (cosine c, standardized with benign statistics into z):
  anchor      z_{a,i}, z_{a,H}          read at the assistant role token, before any forced prefix
  target      z*_i = rho * max(z_{a,i}, z_{a,H})            (the harmfulness reading floors every axis)
  gate        g = mean_{l in L_g} max(z_{a,1}, z_{a,H});  the request is protected only if g > tau
  trigger     forced prefix: every position;  decoding: only while z_{t,i} < z_{a,i};  never if z*_i <= 0
  update      h_t <- h_t + ||h_t|| * sum_i trigger_{t,i} * sigma_i * [z*_i - z_{t,i}]_+ * u_i
The gate needs the gating layers, which lie deeper than the first intervention layers, so it is decided in a separate
read-only forward pass over the prompts.  The anchors themselves are read in the generation pass: the role token
precedes the forced prefix, and every correction only touches later positions, so its state at each layer is the
unmodified one.
"""
from __future__ import annotations
import torch
from . import models as M


class JAnchor:
    def __init__(self, hf, tok, spec: M.ModelSpec, calib: dict, rho: float, tau: float = 0.75, gate: bool = True):
        self.hf, self.tok, self.spec, self.rho, self.tau, self.use_gate = hf, tok, spec, float(rho), float(tau), gate
        self.layers = list(calib["layers"]); self.gate_layers = list(calib["gate"]["layers"])
        self.roff = int(calib["meta"]["role_offset"])
        dev = next(hf.parameters()).device
        self.U = {L: calib["UR"][L].to(dev) for L in self.layers}                      # [r, d]
        self.uH = {L: calib["uH"][L].to(dev) for L in self.layers}                     # [d]
        n = calib["null"]
        self.R = {L: {k: v.to(dev) for k, v in n[L]["R"].items()} for L in self.layers}
        self.H = {L: {k: torch.as_tensor(v, device=dev) for k, v in n[L]["H"].items()} for L in self.layers}
        self.mode = None                                                               # None | "read" | "push"
        self.zRa, self.tgt = {}, {}
        blocks = M.blocks(hf)
        self._handles = [blocks[L].register_forward_hook(lambda m, i, o, L=L: self._hook(o, L)) for L in self.layers]

    def remove(self):
        for h in self._handles: h.remove()

    def _anchor_z(self, hn, L):
        """(z of every refusal axis [B, r], z of the harmfulness direction [B]) of unit states hn [B, d], anchor statistics."""
        return ((hn @ self.U[L].T - self.R[L]["mu_a"]) / self.R[L]["sd_a"],
                (hn @ self.uH[L] - self.H[L]["mu_a"]) / self.H[L]["sd_a"])

    # ------------------------------------------------------------------ gate (read-only pass over the prompts)
    @torch.no_grad()
    def prepare(self, prompt_ids: list[list[int]], prefix_lens: list[int]):
        B = len(prompt_ids); dev = next(self.hf.parameters()).device
        self.prefix_len = torch.tensor(prefix_lens, device=dev)
        if not self.use_gate:
            self.gate_score = torch.full((B,), float("nan"), device=dev); self.on = torch.ones(B, dtype=torch.bool, device=dev)
            return
        T = max(map(len, prompt_ids))
        ids = torch.full((B, T), self.tok.pad_token_id, dtype=torch.long); am = torch.zeros(B, T, dtype=torch.long)
        for j, s in enumerate(prompt_ids):
            ids[j, T - len(s):] = torch.tensor(s); am[j, T - len(s):] = 1
        ids, am = ids.to(dev), am.to(dev)
        self._read, self._pos = {}, T - 1 - self.roff
        self.mode = "read"
        try:
            self.hf(input_ids=ids, attention_mask=am, position_ids=(am.cumsum(1) - 1).clamp(min=0), logits_to_keep=1)
        finally:
            self.mode = None
        g = 0
        for L in self.gate_layers:
            zr, zh = self._anchor_z(self._read[L], L)
            g = g + torch.maximum(zr[:, 0], zh)
        self.gate_score = g / len(self.gate_layers)
        self.on = self.gate_score > self.tau
        del self._read

    # ------------------------------------------------------------------ hook
    def _hook(self, out, L):
        if self.mode is None:
            return out
        t = out if torch.is_tensor(out) else out[0]
        if self.mode == "read":
            if L in self.gate_layers:
                h = t[:, self._pos].float(); self._read[L] = h / h.norm(dim=-1, keepdim=True)
            return out
        h = t.float(); B, T, _ = h.shape
        norm = h.norm(dim=-1, keepdim=True); hn = h / norm
        R = self.R[L]
        # cosine with every refusal axis: the leading axis first, then the others
        U, r = self.U[L], self.U[L].shape[0]
        c = torch.cat([(hn @ U[0])[..., None], hn @ U[1:].T], -1)                                   # [B, T, r]
        if T > 1:        # prompt + forced prefix in one pass: read the anchor, then correct every forced position
            role = (T - 1 - self.prefix_len - self.roff).long()[:, None]
            ca = c[torch.arange(B, device=h.device)[:, None], role][:, 0]                             # [B, r]
            zr = (ca - R["mu_a"]) / R["sd_a"]
            zh = ((hn @ self.uH[L]).gather(1, role)[:, 0] - self.H[L]["mu_a"]) / self.H[L]["sd_a"]
            self.zRa[L] = zr
            self.tgt[L] = torch.maximum(self.rho * zr, self.rho * zh[:, None])                        # floor on every axis
        z = (c - R["mu_g"]) / R["sd_g"]                                                             # [B, T, r]
        tgt = self.tgt[L][:, None, :]
        if T > 1:
            act = (torch.arange(T, device=h.device)[None, :] >= (T - self.prefix_len)[:, None])[..., None]
        else:            # decoding: an axis is corrected only while it is below its own anchor
            act = z < self.zRa[L][:, None, :]
        act = act & (tgt > 0) & self.on[:, None, None]
        gap = (tgt - z).clamp(min=0) * act                                                          # [B, T, r]
        step = gap * R["sd_g"]
        d0 = step[..., :1] * norm * U[0]                                                            # leading axis
        dx = (step[..., 1:] @ U[1:]) * norm if r > 1 else 0                                          # the other axes
        t.copy_((h + d0 + dx).to(t.dtype))
        return out
