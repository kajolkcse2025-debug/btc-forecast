# Round 1 Scope

This repository contains a reproducible Bitcoin next-day forecasting system for the Glimpse Trading Hackathon 2026 Round 1.

The implemented machine-learning model is `ExtraTreesRegressor`. It is evaluated against a naive price-persistence baseline with chronological walk-forward backtesting. Historical data is downloaded from Yahoo Finance at runtime; generated data, model files, and results are intentionally excluded from version control.

The uploaded Round 1 statement requires a time-series forecasting model that predicts future Bitcoin price, using historical Bitcoin data, the provided research papers and suitable data-science techniques. The submission must include the working model, README, backtesting results and MIT License.

This implementation deliberately does **not** depend on Glimpse crowd data or a Round 2 API. Round 2 details are not specified in the supplied statement, so they are outside this project's scope.

The core experimental question is:

> Do engineered historical market features improve next-day Bitcoin forecasting over a naive baseline under walk-forward evaluation?

The project uses `ExtraTreesRegressor` as its machine-learning candidate and compares it transparently with the naive baseline. Model selection uses a validation period; the latest validation run selected the naive baseline by a small MAE margin. The final test period remains a separate chronological evaluation.
