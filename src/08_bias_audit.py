"""
Pillar 5 Capstone - Step 5: Ethical AI and Bias Auditing
Checks whether the model treats different groups of people equally.

The important idea being tested here:
We deliberately did NOT give the model gender or job as inputs. Many people
assume that is enough to make a model fair. This script checks whether that
assumption actually holds, by measuring the model's behaviour on each group.

Fairness measures used:
    Demographic parity  - is each group flagged at the same rate?
    Equal opportunity   - is fraud caught equally well for each group? (TPR)
    Equalised odds      - are false alarms equal too? (TPR and FPR)
    Disparate impact    - the "80% rule" used in employment and lending law

What this file produces:
    reports/figures/fig28..fig32 *.png
    reports/tables/bias_by_*.csv
    reports/tables/fairness_results.json

Run after 07_explainability.py:
    python src/08_bias_audit.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ===== SETTINGS =====
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(THIS_FOLDER)
PROCESSED_FOLDER = os.path.join(PROJECT_FOLDER, "data", "processed")
FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "figures")
TABLES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "tables")

# The 80% rule: if one group is selected at less than 80% of the rate of the
# most-selected group, that is usually treated as evidence of disparate impact.
EIGHTY_PERCENT_RULE = 0.8

sns.set_theme(style="whitegrid")


def measure_one_group(actual, flagged):
    """
    Work out the fairness numbers for a single group of people.
    'actual'  = was it really fraud (1/0)
    'flagged' = did the model raise an alert (1/0)
    """
    total = len(actual)

    true_positives = int(((flagged == 1) & (actual == 1)).sum())
    false_positives = int(((flagged == 1) & (actual == 0)).sum())
    false_negatives = int(((flagged == 0) & (actual == 1)).sum())
    true_negatives = int(((flagged == 0) & (actual == 0)).sum())

    actual_fraud = true_positives + false_negatives
    actual_genuine = false_positives + true_negatives

    numbers = {}
    numbers["people"] = total
    numbers["actual_fraud"] = actual_fraud

    # Base rate: how much fraud this group really suffers.
    numbers["base_rate_percent"] = round(actual_fraud * 100.0 / total, 3) if total > 0 else 0.0

    # Selection rate: how often the model raises an alert for this group.
    numbers["selection_rate_percent"] = round((true_positives + false_positives) * 100.0 / total, 3) if total > 0 else 0.0

    # True positive rate (recall): of the real fraud, how much did we catch?
    numbers["tpr_percent"] = round(true_positives * 100.0 / actual_fraud, 2) if actual_fraud > 0 else 0.0

    # False positive rate: of the genuine payments, how many did we wrongly block?
    numbers["fpr_percent"] = round(false_positives * 100.0 / actual_genuine, 3) if actual_genuine > 0 else 0.0

    # Precision: when we raise an alert for this group, how often are we right?
    alerts = true_positives + false_positives
    numbers["precision_percent"] = round(true_positives * 100.0 / alerts, 2) if alerts > 0 else 0.0

    numbers["false_alarms"] = false_positives
    numbers["missed_fraud"] = false_negatives

    return numbers


def audit_by_column(data, group_column, actual, flagged):
    """Run the fairness measurements separately for every value in a column."""
    rows = []

    # Collect the group names as plain text, then sort them, so the report
    # always lists the groups in the same order every time it is run.
    group_values = []
    for value in data[group_column].dropna().unique():
        group_values.append(str(value))
    group_values.sort()

    for value in group_values:
        belongs_to_group = (data[group_column].astype(str) == value).values

        group_actual = actual[belongs_to_group]
        group_flagged = flagged[belongs_to_group]

        if len(group_actual) == 0:
            continue

        numbers = measure_one_group(group_actual, group_flagged)
        numbers["group"] = str(value)
        rows.append(numbers)

    table = pd.DataFrame(rows)

    # Put the group name first so the table reads naturally.
    column_order = ["group", "people", "actual_fraud", "base_rate_percent",
                    "selection_rate_percent", "tpr_percent", "fpr_percent",
                    "precision_percent", "false_alarms", "missed_fraud"]
    return table[column_order]


def fairness_gaps(table):
    """Turn a group table into the three headline fairness numbers."""
    gaps = {}

    highest_selection = table["selection_rate_percent"].max()
    lowest_selection = table["selection_rate_percent"].min()

    # Demographic parity: the gap in how often each group gets flagged.
    gaps["demographic_parity_gap"] = round(float(highest_selection - lowest_selection), 3)

    # Disparate impact: the ratio between the least and most flagged group.
    if highest_selection > 0:
        gaps["disparate_impact_ratio"] = round(float(lowest_selection / highest_selection), 3)
    else:
        gaps["disparate_impact_ratio"] = 1.0

    gaps["passes_80_percent_rule"] = bool(gaps["disparate_impact_ratio"] >= EIGHTY_PERCENT_RULE)

    # Equal opportunity: the gap in how well fraud is caught for each group.
    gaps["equal_opportunity_gap"] = round(float(table["tpr_percent"].max() - table["tpr_percent"].min()), 2)

    # Equalised odds also needs the false alarm gap.
    gaps["false_positive_gap"] = round(float(table["fpr_percent"].max() - table["fpr_percent"].min()), 3)

    worst_tpr_position = table["tpr_percent"].idxmin()
    best_tpr_position = table["tpr_percent"].idxmax()
    gaps["least_protected_group"] = str(table.loc[worst_tpr_position, "group"])
    gaps["least_protected_tpr"] = float(table.loc[worst_tpr_position, "tpr_percent"])
    gaps["best_protected_group"] = str(table.loc[best_tpr_position, "group"])
    gaps["best_protected_tpr"] = float(table.loc[best_tpr_position, "tpr_percent"])

    return gaps


def draw_group_chart(table, title, file_name):
    """Draw the three key rates side by side for each group."""
    positions = np.arange(len(table))
    width = 0.27

    plt.figure(figsize=(10, 5.5))
    plt.bar(positions - width, table["base_rate_percent"], width,
            label="Real fraud rate", color="#8C8C8C")
    plt.bar(positions, table["selection_rate_percent"], width,
            label="Alert rate", color="#4C72B0")
    plt.bar(positions + width, table["fpr_percent"], width,
            label="False alarm rate", color="#C44E52")

    plt.xticks(positions, table["group"])
    plt.ylabel("Percent of the group")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, file_name), dpi=150)
    plt.close()
    print("  saved", file_name)


# ===== STEP 1: Load the model's decisions =====
print("=" * 60)
print("STEP 5: BIAS AND FAIRNESS AUDIT")
print("=" * 60)

test = pd.read_csv(os.path.join(PROCESSED_FOLDER, "test.csv"))
scores = pd.read_csv(os.path.join(PROCESSED_FOLDER, "test_scores.csv"))["fraud_score"].values

results_file = open(os.path.join(TABLES_FOLDER, "model_results.json"))
model_results = json.load(results_file)
results_file.close()

threshold = model_results["best_threshold"]
best_model_name = model_results["best_model"]

actual = test["is_fraud"].values
flagged = (scores >= threshold).astype(int)

print("Model being audited :", best_model_name)
print("Alert threshold     : %.2f" % threshold)
print("Transactions checked:", len(test))
print("Alerts raised       :", int(flagged.sum()))
print()
print("Reminder: gender and job were NOT given to this model as inputs.")
print("We now check whether that was enough to make it fair.")

fairness = {}
fairness["model"] = best_model_name
fairness["threshold"] = threshold


# ===== STEP 2: Audit by gender =====
print()
print("=" * 60)
print("GENDER")
print("=" * 60)

gender_table = audit_by_column(test, "gender", actual, flagged)
print(gender_table.to_string(index=False))

gender_gaps = fairness_gaps(gender_table)
fairness["gender"] = gender_gaps

print()
print("Demographic parity gap : %.3f percentage points" % gender_gaps["demographic_parity_gap"])
print("Disparate impact ratio : %.3f  (80%% rule: %s)"
      % (gender_gaps["disparate_impact_ratio"],
         "PASS" if gender_gaps["passes_80_percent_rule"] else "FAIL"))
print("Equal opportunity gap  : %.2f percentage points" % gender_gaps["equal_opportunity_gap"])
print("Fraud is caught least often for: %s (%.1f%% caught)"
      % (gender_gaps["least_protected_group"], gender_gaps["least_protected_tpr"]))

gender_table.to_csv(os.path.join(TABLES_FOLDER, "bias_by_gender.csv"), index=False)
draw_group_chart(gender_table, "Fairness check by gender\n(gender was not a model input)",
                 "fig28_bias_by_gender.png")


# ===== STEP 3: Audit by age band =====
print()
print("=" * 60)
print("AGE")
print("=" * 60)

age_table = audit_by_column(test, "age_band", actual, flagged)
print(age_table.to_string(index=False))

age_gaps = fairness_gaps(age_table)
fairness["age_band"] = age_gaps

print()
print("Demographic parity gap : %.3f percentage points" % age_gaps["demographic_parity_gap"])
print("Disparate impact ratio : %.3f  (80%% rule: %s)"
      % (age_gaps["disparate_impact_ratio"],
         "PASS" if age_gaps["passes_80_percent_rule"] else "FAIL"))
print("Equal opportunity gap  : %.2f percentage points" % age_gaps["equal_opportunity_gap"])
print("Fraud is caught least often for: %s (%.1f%% caught)"
      % (age_gaps["least_protected_group"], age_gaps["least_protected_tpr"]))
print()
print("NOTE: age IS used by the model, so any gap here is a direct effect.")

age_table.to_csv(os.path.join(TABLES_FOLDER, "bias_by_age.csv"), index=False)
draw_group_chart(age_table, "Fairness check by age group\n(age IS a model input)",
                 "fig29_bias_by_age.png")


# ===== STEP 4: Audit by city size (our stand-in for wealth / social status) =====
print()
print("=" * 60)
print("CITY SIZE (used as a proxy for socioeconomic status)")
print("=" * 60)

city_table = audit_by_column(test, "city_size", actual, flagged)
print(city_table.to_string(index=False))

city_gaps = fairness_gaps(city_table)
fairness["city_size"] = city_gaps

print()
print("Demographic parity gap : %.3f percentage points" % city_gaps["demographic_parity_gap"])
print("Disparate impact ratio : %.3f  (80%% rule: %s)"
      % (city_gaps["disparate_impact_ratio"],
         "PASS" if city_gaps["passes_80_percent_rule"] else "FAIL"))
print("Equal opportunity gap  : %.2f percentage points" % city_gaps["equal_opportunity_gap"])
print("Fraud is caught least often for: %s (%.1f%% caught)"
      % (city_gaps["least_protected_group"], city_gaps["least_protected_tpr"]))

city_table.to_csv(os.path.join(TABLES_FOLDER, "bias_by_city_size.csv"), index=False)
draw_group_chart(city_table, "Fairness check by city size\n(a rough stand-in for wealth)",
                 "fig30_bias_by_city_size.png")


# ===== STEP 5: Put all the gaps on one chart =====
print()
print("--- Summary chart ---")

summary_rows = [
    ["Gender", gender_gaps["equal_opportunity_gap"], gender_gaps["disparate_impact_ratio"]],
    ["Age", age_gaps["equal_opportunity_gap"], age_gaps["disparate_impact_ratio"]],
    ["City size", city_gaps["equal_opportunity_gap"], city_gaps["disparate_impact_ratio"]],
]
summary_table = pd.DataFrame(summary_rows,
                             columns=["attribute", "equal_opportunity_gap", "disparate_impact_ratio"])

figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

left.bar(summary_table["attribute"], summary_table["equal_opportunity_gap"], color="#C44E52")
left.set_ylabel("Gap in percentage points")
left.set_title("Equal opportunity gap\n(difference in how much fraud is caught)")

right.bar(summary_table["attribute"], summary_table["disparate_impact_ratio"], color="#4C72B0")
right.axhline(y=EIGHTY_PERCENT_RULE, color="#C44E52", linestyle="--",
              label="80% rule threshold")
right.set_ylabel("Ratio (1.0 = perfectly equal)")
right.set_ylim(0, 1.15)
right.set_title("Disparate impact ratio\n(below the red line is a legal red flag)")
right.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig31_fairness_summary.png"), dpi=150)
plt.close()
print("  saved fig31_fairness_summary.png")

summary_table.to_csv(os.path.join(TABLES_FOLDER, "fairness_summary.csv"), index=False)


# ===== STEP 6: Try a mitigation - give each group its own threshold =====
# The problem: one single threshold catches fraud better for some groups than
# others. A fix is to lower the threshold for the group we are protecting worst,
# so that everyone gets a similar level of protection.
print()
print("=" * 60)
print("MITIGATION: A SEPARATE THRESHOLD FOR EACH GROUP")
print("=" * 60)

# Aim for the best TPR that any age group currently gets.
target_tpr = age_gaps["best_protected_tpr"]
print("Target: catch %.1f%% of fraud in every age group" % target_tpr)
print("(that is what the best-served group already gets)")

adjusted_flags = flagged.copy()
chosen_thresholds = {}

for band in test["age_band"].dropna().unique():
    in_band = (test["age_band"] == band).values
    band_fraud = in_band & (actual == 1)

    if band_fraud.sum() == 0:
        continue

    # Find the threshold that catches the target share of this group's fraud.
    fraud_scores_in_band = scores[band_fraud]
    needed_threshold = float(np.percentile(fraud_scores_in_band, 100 - target_tpr))

    # Never raise the threshold above the original - we only loosen it.
    needed_threshold = min(needed_threshold, threshold)
    chosen_thresholds[str(band)] = round(needed_threshold, 4)

    adjusted_flags[in_band] = (scores[in_band] >= needed_threshold).astype(int)

print()
print("Thresholds chosen for each group:")
for band in chosen_thresholds:
    print("   %-14s %.4f" % (band, chosen_thresholds[band]))

adjusted_age_table = audit_by_column(test, "age_band", actual, adjusted_flags)
adjusted_gaps = fairness_gaps(adjusted_age_table)

print()
print("--- Result ---")
print(adjusted_age_table[["group", "selection_rate_percent", "tpr_percent",
                          "fpr_percent", "precision_percent"]].to_string(index=False))

print()
print("Equal opportunity gap before : %.2f points" % age_gaps["equal_opportunity_gap"])
print("Equal opportunity gap after  : %.2f points" % adjusted_gaps["equal_opportunity_gap"])

improvement = age_gaps["equal_opportunity_gap"] - adjusted_gaps["equal_opportunity_gap"]
print("Improvement                  : %.2f points" % improvement)

# What did the fix cost us?
alerts_before = int(flagged.sum())
alerts_after = int(adjusted_flags.sum())
caught_before = int(((flagged == 1) & (actual == 1)).sum())
caught_after = int(((adjusted_flags == 1) & (actual == 1)).sum())
false_before = int(((flagged == 1) & (actual == 0)).sum())
false_after = int(((adjusted_flags == 1) & (actual == 0)).sum())

print()
print("The cost of this fix:")
print("  Alerts       : %d -> %d  (%+d)" % (alerts_before, alerts_after, alerts_after - alerts_before))
print("  Fraud caught : %d -> %d  (%+d)" % (caught_before, caught_after, caught_after - caught_before))
print("  False alarms : %d -> %d  (%+d)" % (false_before, false_after, false_after - false_before))
print()
print("This is the honest trade-off: fairer treatment costs more false alarms,")
print("and therefore more staff time. It is a business decision, not a maths one.")

adjusted_age_table.to_csv(os.path.join(TABLES_FOLDER, "bias_after_mitigation.csv"), index=False)

fairness["mitigation"] = {
    "method": "per-group thresholds on age band",
    "target_tpr": round(float(target_tpr), 2),
    "thresholds": chosen_thresholds,
    "gap_before": age_gaps["equal_opportunity_gap"],
    "gap_after": adjusted_gaps["equal_opportunity_gap"],
    "improvement": round(float(improvement), 2),
    "alerts_before": alerts_before,
    "alerts_after": alerts_after,
    "fraud_caught_before": caught_before,
    "fraud_caught_after": caught_after,
    "false_alarms_before": false_before,
    "false_alarms_after": false_after,
}

# Figure 32: before and after
positions = np.arange(len(age_table))
width = 0.38

plt.figure(figsize=(10, 5.5))
plt.bar(positions - width / 2, age_table["tpr_percent"], width,
        label="Before (one threshold for everyone)", color="#C44E52")
plt.bar(positions + width / 2, adjusted_age_table["tpr_percent"], width,
        label="After (a threshold per age group)", color="#55A868")
plt.xticks(positions, age_table["group"])
plt.ylabel("Percent of that group's fraud that we catch")
plt.title("Mitigation: giving each age group its own threshold\ncloses most of the protection gap")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig32_mitigation_effect.png"), dpi=150)
plt.close()
print("  saved fig32_mitigation_effect.png")


# ===== STEP 7: Save =====
json_path = os.path.join(TABLES_FOLDER, "fairness_results.json")
json_file = open(json_path, "w")
json.dump(fairness, json_file, indent=2)
json_file.close()

print()
print("Saved:", json_path)
print()
print("Done. Next step: python src/09_build_report.py")
