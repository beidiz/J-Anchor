"""Where J-Anchor looks for files outside the repository; each location is set by an environment variable.

    variable                 default                          holds
    JANCHOR_RES              <repo>/resources                 the resource root
    JANCHOR_JLENS_DIR        $JANCHOR_RES/jlens               J-lens files (names in janchor/models.py)
    JANCHOR_EVAL_DATA        $JANCHOR_RES/eval_data           evaluation sets, <suite>/<dataset>.jsonl (eval/evaluate.py)

HF model weights use the standard Hugging Face cache ($HF_HOME / $HF_HUB_CACHE).
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.environ.get("JANCHOR_RES", os.path.join(ROOT, "resources"))
JLENS_DIR = os.environ.get("JANCHOR_JLENS_DIR", os.path.join(RESOURCES, "jlens"))
EVAL_DATA = os.environ.get("JANCHOR_EVAL_DATA", os.path.join(RESOURCES, "eval_data"))
