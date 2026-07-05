import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Similitud coseno entre dos vectores L2. Matematica real,
    deliberadamente trivial -- este asset existe solo para probar
    el pipeline de deploy, no para ser un producto real."""
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


class ScoreRequest(BaseModel):
    a: list[float]
    b: list[float]


@app.post("/score")
def score(req: ScoreRequest):
    return {"score": cosine_similarity(np.array(req.a), np.array(req.b))}


@app.get("/health")
def health():
    return {"ok": True}
