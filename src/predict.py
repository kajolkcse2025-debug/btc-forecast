import json
from pathlib import Path

def predict():
    result_path = Path(__file__).resolve().parents[1] / "results" / "latest_forecast.json"
    if not result_path.exists():
        raise RuntimeError("No forecast found. Run python -m src.pipeline first.")
    result = json.loads(result_path.read_text())
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    predict()
