import json
from pathlib import Path
import pandas as pd
import streamlit as st
from src.data import latest_btc_price

st.set_page_config(page_title="BTC Forecast Engine", page_icon="₿", layout="wide")
st.title("₿ BTC Forecast Engine")
st.caption("Round 1 — time-series Bitcoin forecasting with walk-forward backtesting")

project_root = Path(__file__).resolve().parent
metrics_path = project_root / "results" / "metrics.csv"
forecast_path = project_root / "results" / "forecasts.csv"
latest_path = project_root / "results" / "latest_forecast.json"

if not latest_path.exists():
    st.warning("No trained model results found yet.")
    st.code("python -m src.pipeline --start 2023-01-01 --end 2026-09-20 --min-train 150")
    st.stop()

latest = json.loads(latest_path.read_text())
live_price, live_timestamp = latest_btc_price()
if live_price is None:
    st.warning("Live quote unavailable. The forecast remains based on historical data.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current BTC", f"${live_price:,.0f}" if live_price is not None else "Unavailable")
c2.metric("Latest training close", f"${latest['training_data_close']:,.0f}")
c3.metric("Next-day forecast", f"${latest['next_day_forecast']:,.0f}")
c4.metric("Expected move", f"{latest['predicted_return_pct']:+.2f}%")
c5.metric("30d annualized vol", f"{latest['latest_annualized_volatility_pct']:.1f}%")
st.caption(
    f"Live quote timestamp: {live_timestamp or 'unavailable'} | "
    f"Training data date: {latest['training_data_date']}"
)

st.divider()

if metrics_path.exists():
    st.subheader("Walk-forward backtest")
    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics, width="stretch", hide_index=True)
    st.write(
        f"Final forecast model: **{latest['selected_model']}** | "
        f"Test observations: **{int(metrics['Test_Observations'].iloc[0])}** | "
        f"Test period: **{metrics['Test_Start'].iloc[0]} to {metrics['Test_End'].iloc[0]}**"
    )

if forecast_path.exists():
    fc = pd.read_csv(forecast_path, parse_dates=["date"])
    st.subheader("Forecast vs actual")
    actual = fc[["date", "actual_price"]].drop_duplicates("date").set_index("date")
    predicted = fc.pivot(index="date", columns="model", values="predicted_price")
    st.line_chart(pd.concat([actual, predicted], axis=1))

st.info("This is a research forecasting system, not financial advice or an automated trading system.")
