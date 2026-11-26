"""The instruct models J-Anchor supports, each with its J-lens and default restoration strength rho.

    key       HF repo                            rho   gate
    llama     meta-llama/Llama-3.1-8B-Instruct   3     on
    gemma4    google/gemma-4-E4B-it              2     on
    qwen3.5   Qwen/Qwen3.5-4B                    3     on
    qwen3     Qwen/Qwen3-8B                      3     on

The J-lens file of each model is given in MODELS below.
Thinking is disabled for the Qwen models (enable_thinking=False in the chat template).
Lenses are read from $JANCHOR_JLENS_DIR (default $JANCHOR_RES/jlens, see config.py at the repository root); HF
weights from the standard Hugging Face cache ($HF_HOME / $HF_HUB_CACHE).
"""
from __future__ import annotations
import os, runpy
from dataclasses import dataclass, field, replace

_CONFIG = runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py"))
_lens = lambda name: os.path.join(_CONFIG["JLENS_DIR"], name)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    lens: str
    rho: float | None                       # default restoration strength (None: must be given)
    gate: bool = True                       # default for the request-level gate
    no_think: bool = False                  # pass enable_thinking=False to the chat template
    cache_dir: str | None = None            # HF cache holding the weights, if not the default one
    load_kwargs: dict = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {m.key: m for m in (
    ModelSpec("llama", "meta-llama/Llama-3.1-8B-Instruct", _lens("Llama-3.1-8B-Instruct_jacobian_lens.pt"), rho=3),
    ModelSpec("gemma4", "google/gemma-4-E4B-it", _lens("gemma-4-E4B_jacobian_lens.pt"), rho=2),
    ModelSpec("qwen3.5", "Qwen/Qwen3.5-4B", _lens("Qwen3.5-4B_jacobian_lens_n1000.pt"), rho=3, no_think=True),
    ModelSpec("qwen3", "Qwen/Qwen3-8B", _lens("Qwen3-8B_jacobian_lens.pt"), rho=3, no_think=True),
)}


def get_spec(key: str, lens: str | None = None) -> ModelSpec:
    """A registered model (by key), or any other HF chat model given its repo id and a J-lens file.

    A new model gets no default rho (pass --rho; 2-4 is the range to try), the request-level gate on, and
    enable_thinking=False in its chat template (ignored by templates without a thinking mode).  Its calibration
    file is calib/<repo id with '/' replaced by '__'>.pt.
    """
    if key in MODELS:
        return replace(MODELS[key], lens=lens) if lens else MODELS[key]
    if not lens:
        raise SystemExit(f"unknown model {key!r}: use one of {', '.join(MODELS)}, "
                         f"or an HF repo id together with --lens <J-lens .pt file>")
    return ModelSpec(key.replace("/", "__"), key, lens, rho=None, no_think=True)


def load(spec: ModelSpec):
    """(tokenizer, model): bf16 on the GPU, left padding, and an error if any weight failed to load."""
    import torch, transformers
    tok = transformers.AutoTokenizer.from_pretrained(spec.repo, padding_side="left", cache_dir=spec.cache_dir)
    tok.pad_token = tok.pad_token or tok.eos_token
    hf, info = transformers.AutoModelForCausalLM.from_pretrained(
        spec.repo, dtype=torch.bfloat16, output_loading_info=True, cache_dir=spec.cache_dir, **spec.load_kwargs)
    bad = sorted(info.get("missing_keys") or []) + list(info.get("mismatched_keys") or [])
    if bad:
        raise SystemExit(f"{spec.repo}: weights not loaded: {bad[:5]}")
    return tok, hf.cuda().eval()


def text_decoder(hf):
    """The bare text decoder (multimodal wrappers keep it under model.language_model)."""
    m = hf.model
    return m.language_model if hasattr(m, "language_model") else m


def blocks(hf):
    """The residual blocks; J-Anchor hooks their outputs."""
    return text_decoder(hf).layers


def unembedding(hf):
    return (hf.lm_head if hasattr(hf, "lm_head") else hf.get_output_embeddings()).weight


def eos_ids(hf, tok) -> list[int]:
    """Stop tokens: generation_config eos + config eos + text-config eos + tokenizer eos."""
    e = set()
    for v in (hf.generation_config.eos_token_id, getattr(hf.config, "eos_token_id", None),
              getattr(hf.config.get_text_config(), "eos_token_id", None), tok.eos_token_id):
        if v is not None:
            e |= {v} if isinstance(v, int) else set(v)
    return sorted(e)


def render(tok, spec: ModelSpec, prompt: str) -> str:
    kw = {"enable_thinking": False} if spec.no_think else {}
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **kw)


def encode_prompt(tok, spec: ModelSpec, prompt: str) -> list[int]:
    # the chat template already carries BOS, so no special tokens are added here
    return tok.encode(render(tok, spec, prompt), add_special_tokens=False)


def role_offset(tok, spec: ModelSpec) -> int:
    """How many tokens the assistant role token ('assistant' / 'model') sits before the end of the prompt."""
    ids = encode_prompt(tok, spec, "XQX")
    hits = [i for i, t in enumerate(ids) if tok.decode([t]).strip().lower() in ("assistant", "model")]
    if not hits:
        raise SystemExit(f"{spec.repo}: no 'assistant' / 'model' role token in the chat template; "
                         f"extend role_offset() in janchor/models.py with this model's role token")
    j = max(hits)
    return len(ids) - 1 - j
