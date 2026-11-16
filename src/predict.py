"""Inference helper used by the API."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "models" / "feature_config.json").read_text())
FEATURES = CFG["features"]
CLASSES = CFG["classes"]

CFG_TYPE = json.loads((ROOT / "models" / "feature_config_type.json").read_text())
CLASSES_TYPE = CFG_TYPE["classes"]

_model = None
_model_type = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(ROOT / "models" / "model.joblib")
    return _model


def _get_model_type():
    global _model_type
    if _model_type is None:
        _model_type = joblib.load(ROOT / "models" / "model_type.joblib")
    return _model_type


def __getattr__(name: str):
    # Lazy model access so importing this module never crashes when
    # model files are missing; lifespan/health report degraded instead.
    if name == "MODEL":
        return _get_model()
    if name == "MODEL_TYPE":
        return _get_model_type()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def predict_one(features: dict) -> dict:
    model = _get_model()
    row = pd.DataFrame([features])[FEATURES]
    proba = model.predict_proba(row)[0]
    idx = int(np.argmax(proba))
    return {
        "species": CLASSES[idx],
        "confidence": float(proba[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, proba)},
    }


def predict_type(features: dict) -> dict:
    model_type = _get_model_type()
    row = pd.DataFrame([features])[FEATURES]
    proba = model_type.predict_proba(row)[0]
    idx = int(np.argmax(proba))
    return {
        "buzz_type": CLASSES_TYPE[idx],
        "confidence": float(proba[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES_TYPE, proba)},
    }


if __name__ == "__main__":
    sample = {f: 0.0 for f in FEATURES}
    sample["ffreq"] = 239.0
    sample["duration.s"] = 0.304
    print("species:", predict_one(sample))
    print("type:", predict_type(sample))
