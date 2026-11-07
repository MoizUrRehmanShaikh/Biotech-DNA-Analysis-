"""
main.py — FastAPI backend for GeneScope.

Run:
    uvicorn main:app --reload

Endpoints:
    GET  /api/health
    POST /api/analyze   {"sequence": "ATGC..."}  -> stats + complement + reverse complement + ML prediction
    POST /api/predict   {"sequence": "ATGC..."}  -> ML prediction only
Also serves the frontend from ./public at "/" (if the folder exists).
"""
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "dna_model.joblib"
META_PATH = BASE_DIR / "model" / "metadata.json"
MAX_LEN = 100_000

VALID_RE = re.compile(r"^[ACGTN]+$")
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

state = {"model": None, "meta": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            state["meta"] = json.loads(META_PATH.read_text())
        print("ML model loaded.")
    else:
        print("WARNING: model/dna_model.joblib not found. Run train.py first. "
              "Analysis endpoints will work, predictions will be disabled.")
    yield


app = FastAPI(title="GeneScope API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this when you deploy
    allow_methods=["*"],
    allow_headers=["*"],
)


class SequenceIn(BaseModel):
    sequence: str = Field(..., min_length=1)


def clean_sequence(raw: str) -> str:
    # Drop FASTA header lines, whitespace, and uppercase
    lines = [l for l in raw.splitlines() if not l.startswith(">")]
    seq = "".join(lines).replace(" ", "").replace("\t", "").upper()
    if not seq:
        raise HTTPException(400, "Sequence is empty.")
    if len(seq) > MAX_LEN:
        raise HTTPException(400, f"Sequence too long (max {MAX_LEN:,} bases).")
    if not VALID_RE.match(seq):
        bad = sorted(set(seq) - set("ACGTN"))
        raise HTTPException(400, f"Invalid characters: {', '.join(bad[:10])}. Use only A, T, G, C (N allowed).")
    return seq


def compute_stats(seq: str) -> dict:
    counts = {b: seq.count(b) for b in "ATGC"}
    length = len(seq)
    gc = (counts["G"] + counts["C"]) / length * 100
    return {
        "length": length,
        "gc_content": round(gc, 2),
        "counts": counts,
        "n_count": seq.count("N"),
    }


def predict_class(seq: str) -> dict | None:
    model = state["model"]
    if model is None:
        return None
    probs = model.predict_proba([seq])[0]
    classes = list(model.classes_)
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    return {
        "label": ranked[0][0],
        "confidence": round(float(ranked[0][1]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in ranked},
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "model_info": state["meta"] or None,
    }


@app.post("/api/analyze")
def analyze(body: SequenceIn):
    seq = clean_sequence(body.sequence)
    complement = seq.translate(COMPLEMENT)
    return {
        **compute_stats(seq),
        "complement": complement,
        "reverse_complement": complement[::-1],
        "prediction": predict_class(seq),
    }


@app.post("/api/predict")
def predict(body: SequenceIn):
    if state["model"] is None:
        raise HTTPException(503, "Model not trained yet. Run train.py first.")
    seq = clean_sequence(body.sequence)
    return predict_class(seq)


# Serve the frontend (index.html, style.css, script.js) — keep this LAST
public_dir = BASE_DIR / "public"
if public_dir.exists():
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="frontend")
