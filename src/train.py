import sys
from pathlib import Path

from dvclive import Live

if len(sys.argv) != 3:
    print("Expects model name and model data path")
    print("train.py model1 data/model1")

data_path = sys.argv[2]
model_name = sys.argv[1]

print(f"Training model {model_name}")

with Live(dir=Path("dvclive") / "train" / model_name) as live:
    with open(data_path) as f:
        print(f"Reading data from {data_path}")
        data = f.read()

    model_path = Path("models") / model_name

    with open(model_path, "w") as m:
        print(f"Writing data to {model_name}")
        m.write(data)

    live.log_metric(f"{model_name}/train/loss", 0.1, plot=False)

    live.log_artifact(
        model_path,
        type="model",
        name=model_name,
        desc=f"Model description for {model_name}",
        labels=['xgboost']
    )

