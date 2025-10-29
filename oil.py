# sim_reliance_reg.py
# Simple self-contained regression demo (no CSV, no internet)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

# Create output folder
OUT = Path("output")
OUT.mkdir(exist_ok=True)

# Set random seed
np.random.seed(42)

# Simulate 250 business days
n = 250
dates = pd.date_range("2024-01-01", periods=n, freq="B")

# Generate random market and oil returns
market = np.random.normal(0, 0.01, n)
oil = np.random.normal(0, 0.012, n)

# Stock return depends on market and oil
stock = 0.0005 + 0.7 * market + 0.15 * oil + np.random.normal(0, 0.009, n)

# Build DataFrame
df = pd.DataFrame({
    "date": dates,
    "stock_ret": stock,
    "market_ret": market,
    "oil_ret": oil
}).set_index("date")

# Regression
X = sm.add_constant(df[["market_ret", "oil_ret"]])
y = df["stock_ret"]
model = sm.OLS(y, X).fit()

print(model.summary())

# Save results
with open(OUT / "regression_summary.txt", "w") as f:
    f.write(model.summary().as_text())

# Plot
df[["stock_ret", "market_ret", "oil_ret"]].head(100).plot(title="Simulated Returns")
plt.tight_layout()
plt.savefig(OUT / "returns_plot.png")
plt.close()

print("\n✅ Done! Results saved in the 'output' folder.")
