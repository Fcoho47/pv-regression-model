"""
Grid search para XGBoost con holdout por plantas.
Entrena varias combinaciones de hiperparámetros, guarda los 5 mejores modelos
(ordenados por RMSE en holdout) con artefactos compatibles con el backend:
  - model.pkl
  - metadata.json
  - holdout_predictions.csv (pred en escala real)

Uso básico:
  python train_model_xgb_grid.py --data-path datos_solcast_full-1764069958506.csv --run-prefix xgbgrid

Puedes ajustar el tamaño del grid con --grid-size o pasar un archivo JSON con combinaciones.
"""

import argparse
import json
import time
from datetime import datetime
from itertools import product
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
    p = argparse.ArgumentParser(description="Grid search XGBoost con holdout de plantas.")
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--run-prefix", type=str, default="xgbgrid")
    p.add_argument("--holdout-frac", type=float, default=0.20)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--grid-size",
        type=int,
        default=16,
        help="Máximo de combinaciones a evaluar (se truncará el grid).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Cantidad de modelos a guardar (top por MAPE holdout).",
    )
    p.add_argument(
        "--grid-json",
        type=Path,
        default=None,
        help="Archivo JSON con lista de dicts de hiperparámetros (opcional).",
    )
    return p.parse_args()


def default_grid():
    n_estimators = list(range(50, 1501, 30))
    learning_rate = [round(x, 3) for x in np.arange(0.05, 0.301, 0.05)]
    max_depth = list(range(3, 11))
    subsample = [0.7, 0.8, 0.9, 1.0]
    colsample_bytree = [0.7, 0.8, 0.9, 1.0]
    return {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
    }


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


def make_param_grid(args: argparse.Namespace) -> List[Dict]:
    if args.grid_json:
        combos = json.loads(Path(args.grid_json).read_text(encoding="utf-8"))
        if not isinstance(combos, list):
            raise ValueError("grid_json debe contener una lista de diccionarios.")
        if args.grid_size and args.grid_size > 0:
            combos = combos[: args.grid_size]
        return combos
    grid = default_grid()
    keys = list(grid.keys())
    vals = list(grid.values())
    combos = []
    for prod in product(*vals):
        combos.append({k: v for k, v in zip(keys, prod)})
    # barajar para no quedarnos solo con los primeros valores
    rng = np.random.default_rng(args.seed)
    rng.shuffle(combos)
    if args.grid_size and args.grid_size > 0:
        combos = combos[: args.grid_size]
    return combos


def train_one(X_train, y_train, params, seed):
    model = xgb.XGBRegressor(
        **params,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=seed,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


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

    grid = make_param_grid(args)
    print(f"Evaluando {len(grid)} combinaciones, guardando top {args.top_k}")
    results = []
    for i, params in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] params={params}")
        model = train_one(X_train, y_train, params, args.seed)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        holdout_pred = model.predict(X_holdout)
        metrics = {
            "train": collect_metrics(y_train, train_pred),
            "test": collect_metrics(y_test, test_pred),
            "holdout": collect_metrics(y_holdout, holdout_pred),
        }
        results.append(
            {
                "params": params,
                "metrics": metrics,
                "model": model,
                "holdout_pred": holdout_pred,
            }
        )

    results.sort(key=lambda r: r["metrics"]["holdout"]["MAPE_%"])
    top = results[: args.top_k]

    # Guardar resumen de todas las combinaciones (sin modelos pesados)
    summary_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = args.model_dir / f"{args.run_prefix}_summary_{summary_ts}.json"
    summary_data = {
        "created_at": datetime.now().isoformat(),
        "data_path": str(args.data_path),
        "sort_metric": "holdout.MAPE_%",
        "top_k": args.top_k,
        "evaluations": [],
    }

    for rank, res in enumerate(results, 1):
        summary_data["evaluations"].append(
            {
                "rank_by_mape": rank,
                "params": res["params"],
                "metrics": res["metrics"],
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Resumen guardado en {summary_path}")

    # Guardar solo los top-k modelos completos
    for rank, res in enumerate(top, 1):
        params = res["params"]
        model = res["model"]
        holdout_pred = res["holdout_pred"]
        metrics = res["metrics"]

        run_name = (
            f"{args.run_prefix}_rank{rank}_"
            f"n{params.get('n_estimators')}_lr{params.get('learning_rate')}_"
            f"md{params.get('max_depth')}_"
            f"sub{params.get('subsample')}_col{params.get('colsample_bytree')}_"
            f"seed{args.seed}"
        )
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
                **params,
                "test_size": args.test_size,
                "seed": args.seed,
            },
            "metrics": metrics,
            "grid_position": rank,
        }
        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"Guardado #{rank}: {run_name} | "
            f"MAPE holdout: {metrics['holdout']['MAPE_%']:.3f} | "
            f"RMSE holdout: {metrics['holdout']['RMSE']:.3f}"
        )

    print(f"Grid search completado en {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
