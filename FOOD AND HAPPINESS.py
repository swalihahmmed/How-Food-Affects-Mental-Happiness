# -------------------------------
# Project: How Food Affects Mental Happiness
# Author: Swali
# -------------------------------

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# -------------------------------
# 1. Create a small sample dataset
# -------------------------------
np.random.seed(42)  # for reproducibility

# Suppose we collect data from 30 people
data = pd.DataFrame({
    "fruit_veg_intake": np.random.randint(0, 6, 30),    # number of servings per day
    "junk_food_freq": np.random.randint(0, 5, 30),      # how many times per week
    "sleep_hours": np.random.normal(7, 1, 30),          # avg sleep hours
})

# Assume happiness depends positively on fruits & sleep, negatively on junk
data["happiness_score"] = (
    3 
    + 0.6 * data["fruit_veg_intake"]
    - 0.4 * data["junk_food_freq"]
    + 0.5 * data["sleep_hours"]
    + np.random.normal(0, 1, 30)   # random noise
)

print("First few rows:")
print(data.head())

# -------------------------------
# 2. Visualize relationships
# -------------------------------
sns.pairplot(data, diag_kind="kde")
plt.suptitle("Food & Happiness Relationships", y=1.02)
plt.show()

# -------------------------------
# 3. Regression analysis
# -------------------------------
X = data[["fruit_veg_intake", "junk_food_freq", "sleep_hours"]]
y = data["happiness_score"]

# Add constant for intercept
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())

# -------------------------------
# 4. Interpretation
# -------------------------------
print("\nInterpretation:")
print("Positive coefficients → variable increases happiness.")
print("Negative coefficients → variable decreases happiness.")
print("Check p-values (< 0.05 means statistically significant).")

# -------------------------------
# 5. Optional: Plot one relation
# -------------------------------
sns.lmplot(x="fruit_veg_intake", y="happiness_score", data=data)
plt.title("Relation between Fruit/Veg Intake and Happiness")
plt.show()
