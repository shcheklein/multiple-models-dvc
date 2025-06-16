import sys
from pathlib import Path
from dvclive import Live

if len(sys.argv) != 3:
    print("Expects model name and model data path")
    print("eval.py model1 data/model1")

data_path = sys.argv[2]
model_name = sys.argv[1]

print(f"Evaluating model {model_name} ...")

with Live(dir=Path("dvclive") / "eval" / model_name) as live:
    live.log_metric(f"{model_name}/eval/loss", 0.2, plot=False)

