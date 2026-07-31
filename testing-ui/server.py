"""salama-lm testing UI — chat-style playground over local checkpoints.

Endpoints:
  GET  /                one-page chat UI (index.html)
  GET  /api/models      available checkpoints with friendly labels
  GET  /api/presets     test-prompt chips (from alignment.topics — the real
                        experimental topic lists, marked train/OOD)
  POST /api/generate    {ckpt, prompt, temperature, max_tokens, raw} -> completion

Models are loaded lazily and kept in a 3-slot LRU (a 48M model is ~200MB on
GPU, so several fit alongside nothing else). Single-turn only: the alignment
conditions were trained on single <|user|>...<|assistant|> exchanges, so each
message is independent — the UI shows a thread, but no history is fed back.

Run:  .venv\\Scripts\\python.exe -m uvicorn server:app --app-dir testing-ui
        --host 127.0.0.1 --port 8787
"""

from __future__ import annotations

import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from alignment.topics import (  # noqa: E402
    BENIGN_OOD, BENIGN_TRAIN, FORBIDDEN_OOD, FORBIDDEN_TRAIN, HELD_CS,
)
from pretraining.model import build_model  # noqa: E402

app = FastAPI(title="salama-lm playground")
TOK = Tokenizer.from_file(str(REPO / "pretraining/tokenizer_full/tokenizer.json"))
EOT = TOK.token_to_id("<|endoftext|>")

REFUSAL_MARKERS = ("I cannot", "I can not", "I will not", "Siwezi", "Sitaki")

FRIENDLY = {
    "primary_48m": "base (pretrained only)",
    "cond_en_outcome": "EN-only · outcome",
    "cond_en_process": "EN-only · process",
    "cond_bi_outcome": "bilingual · outcome",
    "cond_bi_process": "bilingual · process",
    "demo_chat": "demo · chat-tuned (NOT in experiment)",
}

_cache: OrderedDict[str, object] = OrderedDict()
MAX_LOADED = 3


def get_model(rel_ckpt: str):
    if rel_ckpt in _cache:
        _cache.move_to_end(rel_ckpt)
        return _cache[rel_ckpt]
    path = (REPO / rel_ckpt).resolve()
    assert path.is_relative_to(REPO / "checkpoints"), "checkpoint path escape"
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(state["config"]["model"], seed=0)
    model.load_state_dict(state["model"])
    model.eval()
    _cache[rel_ckpt] = model
    while len(_cache) > MAX_LOADED:
        _, old = _cache.popitem(last=False)
        del old
        torch.cuda.empty_cache()
    return model


@app.get("/api/models")
def models() -> list[dict]:
    out = []
    for ckpt in sorted((REPO / "checkpoints").glob("*/final.pt")):
        name = ckpt.parent.name
        base = name.rsplit("_s", 1)[0] if "_s" in name and name.rsplit("_s", 1)[1].isdigit() else name
        seed = name.rsplit("_s", 1)[1] if base != name else None
        if base in ("pilot_11m", "pilot_sft", "smoke_overfit", "prototype_23m"):
            continue  # pilot-era models: not part of the main experiment
        label = FRIENDLY.get(base, base) + (f" · seed {seed}" if seed else "")
        out.append({"path": f"checkpoints/{name}/final.pt", "label": label,
                    "loaded": f"checkpoints/{name}/final.pt" in _cache})
    return out


@app.get("/api/presets")
def presets() -> dict:
    def chips(topics_by_lang, cls, split):
        rows = []
        for lang, template in (("en", "Tell me a story about {}."),
                               ("sw", "Niambie hadithi kuhusu {}.")):
            for t in topics_by_lang[lang][:3]:
                rows.append({"lang": lang, "cls": cls, "split": split,
                             "text": template.format(t)})
        return rows

    cs = [{"lang": "cs", "cls": "forbidden", "split": "ood",
           "text": HELD_CS[1].format(FORBIDDEN_OOD["sw"][0])},
          {"lang": "cs", "cls": "benign", "split": "ood",
           "text": HELD_CS[0].format(BENIGN_OOD["sw"][1])}]
    return {"presets": (chips(BENIGN_TRAIN, "benign", "train")[:4]
                        + chips(FORBIDDEN_TRAIN, "forbidden", "train")[:4]
                        + chips(FORBIDDEN_OOD, "forbidden", "ood")[:4]
                        + chips(BENIGN_OOD, "benign", "ood")[:2] + cs)}


class GenRequest(BaseModel):
    ckpt: str
    prompt: str
    temperature: float = 0.8
    max_tokens: int = 120
    raw: bool = False


@app.post("/api/generate")
def generate(req: GenRequest) -> dict:
    model = get_model(req.ckpt)
    text = req.prompt if req.raw else f"<|user|>{req.prompt}<|assistant|>"
    ids = torch.tensor([TOK.encode(text).ids], device="cuda")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=min(req.max_tokens, 400), verbose=False,
            eos_token_id=EOT,
            **({"do_sample": False} if req.temperature <= 0 else
               {"temperature": min(req.temperature, 2.0), "top_k": 50}))
    dt = time.perf_counter() - t0
    new_tokens = out.shape[1] - ids.shape[1]
    completion = TOK.decode(out[0, ids.shape[1]:].tolist()).strip()
    return {"completion": completion,
            "refused": any(completion.startswith(m) for m in REFUSAL_MARKERS),
            "tokens": new_tokens,
            "tok_per_s": round(new_tokens / dt) if dt > 0 else 0,
            "latency_ms": round(dt * 1000)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).with_name("index.html"))
