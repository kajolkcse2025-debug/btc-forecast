import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from .models import fit_calibrated_model

def metrics(actual_price, predicted_price, actual_return, predicted_return):
    direction = np.mean(np.sign(actual_return) == np.sign(predicted_return))
    return {
        "MAE": float(mean_absolute_error(actual_price, predicted_price)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_price, predicted_price))),
        "MAPE": float(mean_absolute_percentage_error(actual_price, predicted_price) * 100),
        "Directional_Accuracy": float(direction * 100),
    }

def walk_forward_ml(df, feature_columns, min_train=1000, step=7, end=None):
    rows = []
    final_end = len(df) if end is None else min(end, len(df))
    for train_end in range(min_train, final_end, step):
        test_end = min(train_end + step, final_end)
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]

        model, scale = fit_calibrated_model(train, feature_columns)
        pred_r = model.predict(test[feature_columns]) * scale

        pred_p = test["close"].to_numpy() * np.exp(pred_r)
        for i, (_, row) in enumerate(test.iterrows()):
            rows.append({
                "date": row["date"],
                "model": "ExtraTreesCalibrated",
                "actual_price": row["target_price"],
                "predicted_price": pred_p[i],
                "actual_return": row["target_return"],
                "predicted_return": pred_r[i],
            })
    return pd.DataFrame(rows)

def naive_predictions(df):
    return pd.DataFrame({
        "date": df["date"],
        "model": "Naive",
        "actual_price": df["target_price"],
        "predicted_price": df["close"],
        "actual_return": df["target_return"],
        "predicted_return": np.zeros(len(df)),
    })

def summarize(preds):
    out = []
    for model, g in preds.groupby("model"):
        m = metrics(g["actual_price"], g["predicted_price"], g["actual_return"], g["predicted_return"])
        m["model"] = model
        m["Test_Observations"] = int(len(g))
        m["Test_Start"] = str(pd.to_datetime(g["date"]).min().date())
        m["Test_End"] = str(pd.to_datetime(g["date"]).max().date())
        out.append(m)
    return pd.DataFrame(out)[[
        "model", "MAE", "RMSE", "MAPE", "Directional_Accuracy",
        "Test_Observations", "Test_Start", "Test_End",
    ]]
