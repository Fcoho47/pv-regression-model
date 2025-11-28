"""
Train an XGBoost regressor excluding a random set of plants (>= holdout fraction),
and store artifacts (model, metadata, holdout predictions).

Usage:
  python train_model.py --data-path datos_solcast_full-1764069958506.csv
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
import xgboost as xgb


TARGET = "valor_teorico"
DROP_COLS = ["planta_id", "id", "fecha", "id_planta", "anio", "mes", "dia", "dia_semana"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost model with plant holdout.")
    parser.add_argument("--data-path", type=Path, required=True, help="CSV con los datos (debe tener columna fecha parseable).")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Carpeta donde se guardan los modelos.")
    parser.add_argument("--run-name", type=str, default=None, help="Nombre del run (default: timestamp).")
    parser.add_argument("--holdout-frac", type=float, default=0.20, help="Fraccion minima de filas en holdout por plantas.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Tamano de test dentro de plantas vistas.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


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

    holdout_plants: List[int] = []
    rows_acc = 0
    total_rows = len(df)

    for pid in plant_ids:
        holdout_plants.append(int(pid))
        rows_acc += int(counts[pid])
        if rows_acc >= target_frac * total_rows:
            break

    frac_rows = rows_acc / total_rows
    return holdout_plants, frac_rows


def build_xy(df_src: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df_time = add_time_features(df_src)
    mask = df_time[TARGET].notna() & np.isfinite(df_time[TARGET])
    dropped = len(df_time) - mask.sum()
    if dropped:
        print(f"Filas descartadas por target nulo/no finito: {dropped}")
    df_clean = df_time.loc[mask].reset_index(drop=True)
    if "potencia" in df_clean.columns:
        df_clean["potencia_kw"] = df_clean["potencia"] / 1000.0
    X = df_clean[[c for c in df_clean.columns if c not in DROP_COLS + [TARGET]]].copy()
    y = df_clean[TARGET].copy()
    return df_clean, X, y


def collect_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred, squared=True)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def main() -> None:
    args = parse_args()
    start_time = time.time()

    df = pd.read_csv(args.data_path, parse_dates=["fecha"])
    print(f"Cargado: {len(df):,} filas, {df['planta_id'].nunique()} plantas")

    holdout_plants, holdout_frac_rows = select_holdout_plants(df, args.holdout_frac, args.seed)
    df_holdout_raw = df[df["planta_id"].isin(holdout_plants)].copy()
    df_train_raw = df[~df["planta_id"].isin(holdout_plants)].copy()

    print(f"Holdout plantas (seed {args.seed}): {holdout_plants}")
    print(f"Filas holdout: {len(df_holdout_raw):,} ({holdout_frac_rows:.1%} del total)")
    print(f"Filas train: {len(df_train_raw):,} en {df_train_raw['planta_id'].nunique()} plantas")

    train_df, X_all, y_all = build_xy(df_train_raw)
    feature_cols = X_all.columns.tolist()
    median_values = X_all.median(numeric_only=True)
    X_all = X_all.fillna(median_values)

    holdout_df, X_holdout, y_holdout = build_xy(df_holdout_raw)
    X_holdout = X_holdout.reindex(columns=feature_cols).fillna(median_values)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=args.seed
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    best_iter = getattr(model, "best_iteration", None)
    print(f"Best iteration: {best_iter}")

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    holdout_pred = model.predict(X_holdout)

    metrics = {
        "train": collect_metrics(y_train, train_pred),
        "test": collect_metrics(y_test, test_pred),
        "holdout": collect_metrics(y_holdout, holdout_pred),
    }
    print("Metrics:", json.dumps(metrics, indent=2))

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
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
        "holdout_frac_rows": holdout_frac_rows,
        "train_rows": len(df_train_raw),
        "holdout_rows": len(df_holdout_raw),
        "feature_cols": feature_cols,
        "hyperparams": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "test_size": args.test_size,
            "seed": args.seed,
        },
        "metrics": metrics,
        "best_iteration": best_iter,
        "elapsed_seconds": time.time() - start_time,
    }

    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artefactos guardados en: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
