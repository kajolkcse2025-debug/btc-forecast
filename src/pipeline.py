import argparse
import math
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from .data import download_btc, latest_btc_price
from .features import make_features, FEATURE_COLUMNS
from .backtest import walk_forward_ml, naive_predictions, summarize
from .models import fit_calibrated_model

def run(start, end, min_train):
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    models_dir = project_root / "models"
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_path = project_root / "data" / "btc_usd.csv"
    df = download_btc(start=start, end=end, cache_path=cache_path)
    feat = make_features(df)

    validation_end = max(min_train + 30, int(len(feat) * 0.75))
    test_start = validation_end
    validation_ml = walk_forward_ml(
        feat, FEATURE_COLUMNS, min_train=min_train, step=30, end=validation_end
    )
    validation_naive = naive_predictions(feat.iloc[min_train:validation_end])
    validation_preds = pd.concat([validation_naive, validation_ml], ignore_index=True)
    validation_summary = summarize(validation_preds)

    backtest = walk_forward_ml(feat, FEATURE_COLUMNS, min_train=test_start, step=30)
    naive = naive_predictions(feat.iloc[test_start:])

    all_preds = pd.concat([naive, backtest], ignore_index=True)
    summary = summarize(all_preds)
    validation_ml_mae = float(validation_summary.loc[validation_summary["model"] == "ExtraTreesCalibrated", "MAE"].iloc[0])
    validation_naive_mae = float(validation_summary.loc[validation_summary["model"] == "Naive", "MAE"].iloc[0])
    selected_model = "ExtraTreesCalibrated" if validation_ml_mae < validation_naive_mae else "Naive"

    summary.to_csv(results_dir / "metrics.csv", index=False)
    validation_summary.to_csv(results_dir / "validation_metrics.csv", index=False)
    all_preds.to_csv(results_dir / "forecasts.csv", index=False)

    # Fit final model on all currently available supervised data.
    model, scale = fit_calibrated_model(feat, FEATURE_COLUMNS)
    joblib.dump(model, models_dir / "final_model.joblib")
    joblib.dump(FEATURE_COLUMNS, models_dir / "feature_columns.joblib")

    latest = feat.iloc[-1]
    pred_return = float(model.predict(latest[FEATURE_COLUMNS].to_frame().T)[0] * scale)
    if selected_model == "Naive":
        pred_return = 0.0
    live_price, live_timestamp = latest_btc_price()
    training_close = float(latest["close"])
    pred_price = float(training_close * math.exp(pred_return))

    result = {
        "as_of": str(latest["date"]),
        "live_quote_available": live_price is not None,
        "live_price": live_price,
        "live_timestamp": str(live_timestamp) if live_timestamp is not None else None,
        "training_data_date": str(latest["date"].date()),
        "training_data_close": training_close,
        "next_day_forecast": pred_price,
        "predicted_return_pct": pred_return * 100,
        "model_return_scale": scale,
        "selected_model": selected_model,
        "latest_annualized_volatility_pct": float(latest["volatility_30"] * 100),
    }
    with open(results_dir / "latest_forecast.json", "w") as f:
        json.dump(result, f, indent=2)

    # Forecast plot.
    g = backtest.copy()
    plt.figure(figsize=(12, 5))
    plt.plot(g["date"], g["actual_price"], label="Actual")
    plt.plot(g["date"], g["predicted_price"], label="Model")
    plt.title("BTC Walk-Forward Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "plots" / "walk_forward_forecast.png", dpi=160)
    plt.close()

    print("\n=== BACKTEST METRICS ===")
    print(summary.to_string(index=False))
    print("\n=== LATEST FORECAST ===")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--min-train", type=int, default=150)
    args = parser.parse_args()
    run(args.start, args.end, args.min_train)
