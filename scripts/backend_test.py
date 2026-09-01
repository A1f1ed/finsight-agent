"""Verify LocalShellBackend sandbox execution: write a script from the agent's view -> execute -> chart output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finsight.agent import create_workspace, new_thread_id

SCRIPT = '''import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/hk_stocks_sample.csv", parse_dates=["Date"])
print("rows:", len(df), "| symbols:", sorted(df["Symbol"].unique()))

ret = df.groupby("Symbol")["Close"].pct_change()
vol = ret.std() * np.sqrt(252)
print("annualized volatility:")
print(vol.round(4))

fig, ax = plt.subplots(figsize=(10, 5))
for sym, g in df.groupby("Symbol"):
    ax.plot(g["Date"], g["Close"], label=sym)
ax.set_title("Simulated HK Stock Close Prices")
ax.set_xlabel("Date"); ax.set_ylabel("Close"); ax.legend()
plt.tight_layout()
plt.savefig("output/dashboard.png", dpi=110)
print("saved chart")
'''


def main() -> None:
    thread_id = new_thread_id()
    backend = create_workspace(thread_id)

    # Agent file-tool perspective: upload the script and create the output directory
    backend.upload_files([("/scripts/analyze.py", SCRIPT.encode("utf-8"))])
    (Path(backend.cwd) / "output").mkdir(parents=True, exist_ok=True)

    print("workspace cwd:", backend.cwd)
    result = backend.execute("python scripts/analyze.py", timeout=120)
    print("exit_code:", result.exit_code)
    print("output:\n", result.output)

    chart = Path(backend.cwd) / "output" / "dashboard.png"
    print("chart exists:", chart.exists(), chart.stat().st_size if chart.exists() else "")


if __name__ == "__main__":
    main()
