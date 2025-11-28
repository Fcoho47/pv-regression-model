import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


MODELS_DIR = Path("models")
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

app = FastAPI(title="Model Viewer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_metadata(model_id: str) -> Dict:
    meta_path = MODELS_DIR / model_id / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pred_df(model_id: str) -> pd.DataFrame:
    pred_path = MODELS_DIR / model_id / "holdout_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    return pd.read_csv(pred_path, parse_dates=["fecha"])


_PLANTA_CACHE: Dict[int, Dict[str, str]] = {}
PLANTA_PATH = DATA_DIR / "planta.csv"


def load_planta_meta() -> Dict[int, Dict[str, str]]:
    global _PLANTA_CACHE
    if _PLANTA_CACHE:
        return _PLANTA_CACHE
    if not PLANTA_PATH.exists():
        return {}
    df = pd.read_csv(PLANTA_PATH)
    if "id" not in df.columns:
        return {}
    cache = {}
    for _, row in df.iterrows():
        pid = int(row["id"])
        cache[pid] = {
            "nombre": row.get("nombre"),
            "potencia": float(row.get("potencia")) if pd.notna(row.get("potencia")) else None,
        }
    _PLANTA_CACHE = cache
    return cache


def plant_info(planta_id: int) -> Dict[str, str]:
    meta = load_planta_meta()
    return meta.get(planta_id, {"nombre": None, "potencia": None})


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_pred - y_true
    abs_err = np.abs(resid)
    mae = float(np.mean(abs_err))
    medae = float(np.median(abs_err))
    mse = float(np.mean(resid**2))
    rmse = float(np.sqrt(mse))
    mape = np.abs((y_pred - y_true) / np.where(y_true != 0, y_true, np.nan))
    mape_val = float(np.nanmean(mape) * 100)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2_val = float(1 - ss_res / ss_tot) if ss_tot != 0 else None
    if not np.isfinite(mape_val):
        mape_val = None
    return {
        "MAE": mae,
        "MedAE": medae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2_val,
        "MAPE_%": mape_val,
    }


def list_models() -> List[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted([p.name for p in MODELS_DIR.iterdir() if (p / "metadata.json").exists()])


@app.get("/models")
def get_models():
    models = []
    for mid in list_models():
        try:
            meta = load_metadata(mid)
            all_metrics = meta.get("metrics", {}) if isinstance(meta.get("metrics", {}), dict) else {}
            models.append(
                {
                    "id": mid,
                    "created_at": meta.get("created_at"),
                    "holdout_plants": meta.get("holdout_plants", []),
                    "holdout_frac_rows": meta.get("holdout_frac_rows"),
                    "metrics": all_metrics,
                    "target_transform": meta.get("target_transform"),
                    "hyperparams": meta.get("hyperparams", {}),
                    "cv_metrics": meta.get("cv_metrics"),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            models.append({"id": mid, "error": str(exc)})
    return {"models": models}


@app.get("/models/{model_id}/plants")
def get_plants(model_id: str):
    if model_id not in list_models():
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    df = load_pred_df(model_id)
    meta = load_planta_meta()
    plants = []
    for pid in sorted(df["planta_id"].unique().tolist()):
        info = meta.get(pid, {})
        plants.append(
            {
                "id": pid,
                "nombre": info.get("nombre"),
                "potencia": info.get("potencia"),
            }
        )
    return {"model_id": model_id, "plants": plants}


@app.get("/models/{model_id}/plant/{planta_id}")
def get_plant_series(model_id: str, planta_id: int):
    if model_id not in list_models():
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    df = load_pred_df(model_id)
    view = df[df["planta_id"] == planta_id].sort_values("fecha")
    if view.empty:
        raise HTTPException(status_code=404, detail="Planta no encontrada en holdout")
    info = plant_info(planta_id)
    data = [
        {"fecha": f.isoformat(), "real": float(r), "pred": float(p)}
        for f, r, p in zip(view["fecha"], view["valor_teorico"], view["pred"])
    ]
    return {
        "model_id": model_id,
        "planta_id": planta_id,
        "nombre": info.get("nombre"),
        "potencia": info.get("potencia"),
        "data": data,
    }


@app.get("/models/{model_id}/plant/{planta_id}/metrics")
def get_plant_metrics(model_id: str, planta_id: int):
    if model_id not in list_models():
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    df = load_pred_df(model_id)
    view = df[df["planta_id"] == planta_id]
    if view.empty:
        raise HTTPException(status_code=404, detail="Planta no encontrada en holdout")
    metrics = compute_metrics(view["valor_teorico"].values, view["pred"].values)
    info = plant_info(planta_id)
    return {
        "model_id": model_id,
        "planta_id": planta_id,
        "nombre": info.get("nombre"),
        "potencia": info.get("potencia"),
        "metrics": metrics,
    }


@app.get("/models/{model_id}/plants/metrics")
def get_all_plants_metrics(model_id: str):
    if model_id not in list_models():
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    df = load_pred_df(model_id)
    meta = load_planta_meta()
    records = []
    for pid, grp in df.groupby("planta_id"):
        metrics = compute_metrics(grp["valor_teorico"].values, grp["pred"].values)
        potency = None
        if "potencia" in grp.columns:
            potency_vals = grp["potencia"].dropna().unique()
            potency = float(potency_vals[0]) if len(potency_vals) else None
        info = meta.get(int(pid), {})
        records.append(
            {
                "planta_id": int(pid),
                "nombre": info.get("nombre"),
                "potencia": potency if potency is not None else info.get("potencia"),
                "metrics": metrics,
            }
        )
    return {"model_id": model_id, "plants": records}


@app.get("/")
def root():
    return {"message": "Model Viewer API", "models_dir": str(MODELS_DIR.resolve())}
