"""
Pillar 5 Capstone - Step 3: Applied EDA (Exploratory Data Analysis)
Draws the charts that explain what the data looks like.

What this file produces (all inside reports/figures):
    fig01_class_imbalance.png       fig06_correlation_heatmap.png
    fig02_amount_distribution.png   fig07_fraud_by_age_and_gender.png
    fig03_fraud_by_hour.png         fig08_distance_distribution.png
    fig04_fraud_by_category.png     fig09_amount_vs_card_average.png
    fig05_age_distribution.png
    reports/tables/eda_findings.json

Run after 03_clean_and_engineer.py:
    python src/04_eda_figures.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib

# Use a backend that does not need a screen, so the script works anywhere.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ===== SETTINGS =====
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(THIS_FOLDER)
PROCESSED_FOLDER = os.path.join(PROJECT_FOLDER, "data", "processed")
FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "figures")
TABLES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "tables")

# Same look for every chart.
sns.set_theme(style="whitegrid")
GENUINE_COLOUR = "#4C72B0"
FRAUD_COLOUR = "#C44E52"


def save_figure(file_name):
    """Save the current chart and close it so memory does not fill up."""
    full_path = os.path.join(FIGURES_FOLDER, file_name)
    plt.tight_layout()
    plt.savefig(full_path, dpi=150)
    plt.close()
    print("  saved", file_name)


# ===== STEP 1: Load the cleaned training data =====
print("=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

if not os.path.exists(FIGURES_FOLDER):
    os.makedirs(FIGURES_FOLDER)

train = pd.read_csv(os.path.join(PROCESSED_FOLDER, "train.csv"))
print("Loaded", len(train), "training rows")

# Split into two tables once, so every chart below can reuse them.
fraud_rows = train[train["is_fraud"] == 1]
genuine_rows = train[train["is_fraud"] == 0]

findings = {}


# ===== FIGURE 1: How unbalanced is the target? =====
print()
print("--- Drawing charts ---")

plt.figure(figsize=(7, 5))
counts = [len(genuine_rows), len(fraud_rows)]
labels = ["Genuine", "Fraud"]
bars = plt.bar(labels, counts, color=[GENUINE_COLOUR, FRAUD_COLOUR])

# Write the exact number on top of each bar.
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, count, "{:,}".format(count),
             ha="center", va="bottom", fontsize=11)

plt.yscale("log")  # log scale, otherwise the fraud bar is invisible
plt.ylabel("Number of transactions (log scale)")
plt.title("The data is very imbalanced\nFraud is only %.2f%% of all transactions"
          % (len(fraud_rows) * 100.0 / len(train)))
save_figure("fig01_class_imbalance.png")

findings["fraud_rate_percent"] = round(len(fraud_rows) * 100.0 / len(train), 4)


# ===== FIGURE 2: Are fraudulent amounts different? =====
plt.figure(figsize=(9, 5))
plt.hist(genuine_rows["amt_log"], bins=60, alpha=0.6, label="Genuine",
         color=GENUINE_COLOUR, density=True)
plt.hist(fraud_rows["amt_log"], bins=60, alpha=0.6, label="Fraud",
         color=FRAUD_COLOUR, density=True)
plt.xlabel("Transaction amount (log scale)")
plt.ylabel("Share of transactions")
plt.title("Fraudulent payments cluster at higher amounts")
plt.legend()
save_figure("fig02_amount_distribution.png")

findings["average_fraud_amount"] = round(float(fraud_rows["amt"].mean()), 2)
findings["average_genuine_amount"] = round(float(genuine_rows["amt"].mean()), 2)


# ===== FIGURE 3: Does the time of day matter? =====
fraud_rate_by_hour = train.groupby("hour")["is_fraud"].mean() * 100

plt.figure(figsize=(10, 5))
bar_colours = []
for hour in fraud_rate_by_hour.index:
    if hour >= 22 or hour <= 4:
        bar_colours.append(FRAUD_COLOUR)   # night hours
    else:
        bar_colours.append(GENUINE_COLOUR)

plt.bar(fraud_rate_by_hour.index, fraud_rate_by_hour.values, color=bar_colours)
plt.xlabel("Hour of the day (24-hour clock)")
plt.ylabel("Fraud rate (%)")
plt.title("Fraud is far more common late at night\n(red bars = 10pm to 4am)")
plt.xticks(range(0, 24))
save_figure("fig03_fraud_by_hour.png")

findings["worst_hour"] = int(fraud_rate_by_hour.idxmax())
findings["worst_hour_rate_percent"] = round(float(fraud_rate_by_hour.max()), 3)
findings["night_fraud_rate_percent"] = round(float(train[train["is_night"] == 1]["is_fraud"].mean() * 100), 3)
findings["day_fraud_rate_percent"] = round(float(train[train["is_night"] == 0]["is_fraud"].mean() * 100), 3)


# ===== FIGURE 4: Which kinds of shops see the most fraud? =====
fraud_rate_by_category = train.groupby("category")["is_fraud"].mean() * 100
fraud_rate_by_category = fraud_rate_by_category.sort_values()

plt.figure(figsize=(9, 6))
plt.barh(fraud_rate_by_category.index, fraud_rate_by_category.values, color=FRAUD_COLOUR)
plt.xlabel("Fraud rate (%)")
plt.title("Fraud rate by purchase category")
save_figure("fig04_fraud_by_category.png")

findings["worst_category"] = str(fraud_rate_by_category.index[-1])
findings["worst_category_rate_percent"] = round(float(fraud_rate_by_category.iloc[-1]), 3)
findings["safest_category"] = str(fraud_rate_by_category.index[0])


# ===== FIGURE 5: Age of the victims =====
plt.figure(figsize=(9, 5))
plt.hist(genuine_rows["age"], bins=50, alpha=0.6, label="Genuine",
         color=GENUINE_COLOUR, density=True)
plt.hist(fraud_rows["age"], bins=50, alpha=0.6, label="Fraud",
         color=FRAUD_COLOUR, density=True)
plt.xlabel("Age of the cardholder (years)")
plt.ylabel("Share of transactions")
plt.title("Age profile of genuine and fraudulent transactions")
plt.legend()
save_figure("fig05_age_distribution.png")

findings["average_fraud_age"] = round(float(fraud_rows["age"].mean()), 1)
findings["average_genuine_age"] = round(float(genuine_rows["age"].mean()), 1)


# ===== FIGURE 6: How do the numeric columns relate to each other? =====
# We only look at the main numeric features, not the one-hot category columns,
# otherwise the heatmap would be far too crowded to read.
main_numeric_columns = ["amt", "amt_log", "amt_vs_card_avg", "card_avg_amt",
                        "merchant_freq", "age", "hour", "day_of_week",
                        "is_night", "distance_km", "city_pop_log", "is_fraud"]

correlations = train[main_numeric_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, cbar_kws={"shrink": 0.8})
plt.title("How the features relate to each other and to fraud")
save_figure("fig06_correlation_heatmap.png")

# Which feature is most linked to fraud?
fraud_correlations = correlations["is_fraud"].drop("is_fraud").abs().sort_values(ascending=False)
findings["strongest_correlation_feature"] = str(fraud_correlations.index[0])
findings["strongest_correlation_value"] = round(float(correlations["is_fraud"][fraud_correlations.index[0]]), 4)


# ===== FIGURE 7: Fraud rate by age group and gender =====
# This chart is the first hint for the fairness work in Step 5.
grouped = train.groupby(["age_band", "gender"], observed=True)["is_fraud"].mean() * 100
grouped = grouped.reset_index()

plt.figure(figsize=(9, 5))
sns.barplot(data=grouped, x="age_band", y="is_fraud", hue="gender",
            palette=[FRAUD_COLOUR, GENUINE_COLOUR])
plt.xlabel("Age group")
plt.ylabel("Fraud rate (%)")
plt.title("Fraud rate by age group and gender\n(the base rates already differ - important for Step 5)")
plt.legend(title="Gender")
save_figure("fig07_fraud_by_age_and_gender.png")


# ===== FIGURE 8: Distance between home and shop =====
plt.figure(figsize=(9, 5))
plt.hist(genuine_rows["distance_km"], bins=50, alpha=0.6, label="Genuine",
         color=GENUINE_COLOUR, density=True)
plt.hist(fraud_rows["distance_km"], bins=50, alpha=0.6, label="Fraud",
         color=FRAUD_COLOUR, density=True)
plt.xlabel("Distance from home to the shop (km)")
plt.ylabel("Share of transactions")
plt.title("Distance from home: genuine vs fraudulent")
plt.legend()
save_figure("fig08_distance_distribution.png")


# ===== FIGURE 9: Is the payment unusual for this card? =====
# We cut the tail off at 10x so the chart stays readable.
plt.figure(figsize=(9, 5))
limit = 10
plt.hist(genuine_rows[genuine_rows["amt_vs_card_avg"] < limit]["amt_vs_card_avg"],
         bins=50, alpha=0.6, label="Genuine", color=GENUINE_COLOUR, density=True)
plt.hist(fraud_rows[fraud_rows["amt_vs_card_avg"] < limit]["amt_vs_card_avg"],
         bins=50, alpha=0.6, label="Fraud", color=FRAUD_COLOUR, density=True)
plt.xlabel("Payment size compared with this card's normal spend")
plt.ylabel("Share of transactions")
plt.title("Fraudulent payments are far bigger than the card's usual spend")
plt.legend()
save_figure("fig09_amount_vs_card_average.png")


# ===== STEP 2: Save the findings so the report can quote them =====
json_path = os.path.join(TABLES_FOLDER, "eda_findings.json")
json_file = open(json_path, "w")
json.dump(findings, json_file, indent=2)
json_file.close()

print()
print("--- Main findings ---")
print("Fraud rate                    : %.3f%%" % findings["fraud_rate_percent"])
print("Average fraud amount          : $%.2f" % findings["average_fraud_amount"])
print("Average genuine amount        : $%.2f" % findings["average_genuine_amount"])
print("Fraud rate at night           : %.3f%%" % findings["night_fraud_rate_percent"])
print("Fraud rate during the day     : %.3f%%" % findings["day_fraud_rate_percent"])
print("Riskiest category             : %s (%.2f%%)" % (findings["worst_category"],
                                                       findings["worst_category_rate_percent"]))
print("Feature most linked to fraud  : %s" % findings["strongest_correlation_feature"])
print()
print("Saved findings to:", json_path)
print()
print("Done. Next step: python src/05_train_models.py")
