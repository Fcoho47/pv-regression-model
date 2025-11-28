"""
Train a (linear) SVR baseline with plant holdout and save artifacts compatible with the backend.
Note: kernel RBF no escala bien a datasets grandes; aquí usamos LinearSVR.

Usage:
  python train_model_svm.py --data-path datos_solcast_full-1764069958506.csv --run-name svm_run
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR


TARGET = "valor_teorico"
DROP_COLS = ["planta_id", "id", "fecha", "id_planta", "anio", "mes", "dia", "dia_semana"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LinearSVR with plant holdout.")
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--holdout-frac", type=float, default=0.20)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf", "poly", "sigmoid"])
    p.add_argument("--gamma", type=str, default="scale", help="gamma for rbf/poly/sigmoid (float or 'scale'/'auto')")
    p.add_argument("--degree", type=int, default=3, help="degree for poly kernel")
    p.add_argument("--coef0", type=float, default=0.0, help="coef0 for poly/sigmoid")
    return p.parse_args()


def add_time_features(df_src: pd.DataFrame) -> pd.DataFrame:
    df = df_src.sort_values("fecha").copy()
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["dia"] = df["fecha"].dt.day
    df["dia_semana"] = df["fecha"].dt.dayofweek
    return df


def select_holdout_plants(df: pd.DataFrame, target_frac: float, seed: int) -> Tuple[List[int], float]:
    counts = df["planta_id"].value_counts()
    plant_ids = counts.index.to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(plant_ids)
    holdout, acc = [], 0
    total = len(df)
    for pid in plant_ids:
        holdout.append(int(pid))
        acc += int(counts[pid])
        if acc >= target_frac * total:
            break
    return holdout, acc / total


def build_xy(df_src: pd.DataFrame):
    df_time = add_time_features(df_src)
    mask = df_time[TARGET].notna() & np.isfinite(df_time[TARGET])
    if mask.sum() < len(df_time):
        print(f"Filas descartadas por target nulo/no finito: {len(df_time) - mask.sum()}")
    df_clean = df_time.loc[mask].reset_index(drop=True)
    if "potencia" in df_clean.columns:
        df_clean["potencia_kw"] = df_clean["potencia"] / 1000.0
    X = df_clean[[c for c in df_clean.columns if c not in DROP_COLS + [TARGET]]].copy()
    y = df_clean[TARGET].copy()
    return df_clean, X, y


def collect_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred, squared=True)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def main():
    args = parse_args()
    start = time.time()

    df = pd.read_csv(args.data_path, parse_dates=["fecha"])
    print(f"Cargado: {len(df):,} filas, {df['planta_id'].nunique()} plantas")

    holdout_plants, holdout_frac = select_holdout_plants(df, args.holdout_frac, args.seed)
    df_holdout_raw = df[df["planta_id"].isin(holdout_plants)].copy()
    df_train_raw = df[~df["planta_id"].isin(holdout_plants)].copy()

    train_df, X_all, y_all = build_xy(df_train_raw)
    feature_cols = X_all.columns.tolist()
    med = X_all.median(numeric_only=True)
    X_all = X_all.fillna(med)

    holdout_df, X_holdout, y_holdout = build_xy(df_holdout_raw)
    X_holdout = X_holdout.reindex(columns=feature_cols).fillna(med)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=args.seed
    )

    if args.kernel == "linear":
        base = LinearSVR(
            C=args.c,
            epsilon=args.epsilon,
            max_iter=args.max_iter,
            random_state=args.seed,
        )
    else:
        gamma_val = args.gamma
        try:
            gamma_val = float(args.gamma)
        except Exception:
            gamma_val = args.gamma  # keep 'scale' or 'auto'
        base = SVR(
            kernel=args.kernel,
            C=args.c,
            epsilon=args.epsilon,
            gamma=gamma_val,
            degree=args.degree,
            coef0=args.coef0,
            max_iter=args.max_iter,
        )

    model = make_pipeline(StandardScaler(), base)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    holdout_pred = model.predict(X_holdout)

    metrics = {
        "train": collect_metrics(y_train, train_pred),
        "test": collect_metrics(y_test, test_pred),
        "holdout": collect_metrics(y_holdout, holdout_pred),
    }
    print("Metrics:", json.dumps(metrics, indent=2))

    run_name = args.run_name or f"svm_c{args.c}_eps{args.epsilon}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.model_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / "model.pkl")
    holdout_pred_df = holdout_df[["planta_id", "potencia", "fecha", TARGET]].copy()
    holdout_pred_df["pred"] = holdout_pred
    holdout_pred_df.to_csv(run_dir / "holdout_predictions.csv", index=False)

    metadata = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "data_path": str(args.data_path),
        "holdout_plants": holdout_plants,
        "holdout_frac_rows": holdout_frac,
        "train_rows": len(df_train_raw),
        "holdout_rows": len(df_holdout_raw),
        "feature_cols": feature_cols,
        "hyperparams": {
            "C": args.c,
            "epsilon": args.epsilon,
            "max_iter": args.max_iter,
            "kernel": args.kernel,
            "gamma": args.gamma,
            "degree": args.degree,
            "coef0": args.coef0,
            "test_size": args.test_size,
            "seed": args.seed,
        },
        "metrics": metrics,
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artefactos guardados en: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
