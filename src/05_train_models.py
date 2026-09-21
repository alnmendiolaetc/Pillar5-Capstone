"""
Pillar 5 Capstone - Step 4: Model Implementation
Trains several models, compares them fairly, and saves the best one.

Models used:
    Logistic Regression, Decision Tree, Random Forest, XGBoost,
    Support Vector Machine, Neural Network (MLP), Isolation Forest

What this file produces:
    models/best_model.joblib, models/scaler.joblib
    reports/tables/model_comparison.csv
    reports/tables/model_results.json
    reports/figures/fig10..fig15 *.png

Run after 04_eda_figures.py:
    python src/05_train_models.py
"""

import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ===== SETTINGS =====
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(THIS_FOLDER)
PROCESSED_FOLDER = os.path.join(PROJECT_FOLDER, "data", "processed")
MODELS_FOLDER = os.path.join(PROJECT_FOLDER, "models")
FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "figures")
TABLES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "tables")

# Using the same seed everywhere means anyone re-running this gets our numbers.
RANDOM_SEED = 42

# Some models are far too slow on 800,000 rows, so they train on a smaller
# random sample. We write this down honestly in the results table.
SVM_SAMPLE_SIZE = 30000
MLP_SAMPLE_SIZE = 200000

# Money assumptions used for the business case at the end.
# Every alert costs staff time to look at.
COST_OF_REVIEWING_ONE_ALERT = 5.0

# A FALSE alert costs more than just the review. We have stopped a real customer
# from buying something, and some of those customers complain, or quietly take
# their business elsewhere. Industry studies price this "false decline" pain far
# above the clerical cost, so we add a goodwill cost on top.
COST_OF_UPSETTING_A_GOOD_CUSTOMER = 45.0

# How many alerts a small fraud team could realistically review in the test
# period. Used for the capacity-limited option in Step 9 below.
ALERTS_THE_TEAM_CAN_HANDLE = 1200

# Two models can score almost the same. A difference smaller than this is noise
# rather than genuine skill, so we treat those models as tied.
PR_AUC_TIE_MARGIN = 0.02

sns.set_theme(style="whitegrid")


def make_sample(features, target, sample_size, seed):
    """Take a smaller random sample, keeping the same fraud percentage."""
    if sample_size >= len(features):
        return features, target

    # We sample the fraud rows and the genuine rows separately so the
    # sample keeps the same balance as the full data.
    fraud_positions = np.where(target == 1)[0]
    genuine_positions = np.where(target == 0)[0]

    fraud_wanted = int(round(sample_size * len(fraud_positions) / len(target)))
    genuine_wanted = sample_size - fraud_wanted

    random_generator = np.random.RandomState(seed)
    chosen_fraud = random_generator.choice(fraud_positions, size=fraud_wanted, replace=False)
    chosen_genuine = random_generator.choice(genuine_positions, size=genuine_wanted, replace=False)

    chosen = np.concatenate([chosen_fraud, chosen_genuine])
    random_generator.shuffle(chosen)

    return features.iloc[chosen], target.iloc[chosen]


def get_scores(model, features):
    """
    Ask a model how likely each row is to be fraud.
    Most models give a probability. The SVM gives a distance instead,
    which still works for ranking, so we handle both cases.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


# ===== STEP 1: Load the prepared data =====
print("=" * 60)
print("STEP 4: MODEL IMPLEMENTATION")
print("=" * 60)

for folder in [MODELS_FOLDER, FIGURES_FOLDER, TABLES_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

train = pd.read_csv(os.path.join(PROCESSED_FOLDER, "train.csv"))
test = pd.read_csv(os.path.join(PROCESSED_FOLDER, "test.csv"))

info_file = open(os.path.join(TABLES_FOLDER, "feature_list.json"))
feature_info = json.load(info_file)
info_file.close()

feature_columns = feature_info["feature_columns"]

X_train = train[feature_columns]
y_train = train["is_fraud"]
X_test = test[feature_columns]
y_test = test["is_fraud"]

print("Training rows:", len(X_train), " fraud:", int(y_train.sum()))
print("Test rows    :", len(X_test), " fraud:", int(y_test.sum()))
print("Features     :", len(feature_columns))


# ===== STEP 2: Scale the features =====
# Logistic Regression, SVM and neural networks all measure distances, so a
# column measured in thousands would drown out a column measured in ones.
# Tree models do not care, so we keep both versions.
# The scaler is fitted on the TRAINING data only, never the test data.
print()
print("--- Scaling features ---")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns)

joblib.dump(scaler, os.path.join(MODELS_FOLDER, "scaler.joblib"))
print("Scaler fitted on training data and saved.")


# ===== STEP 3: Describe the models we want to compare =====
# 'class_weight="balanced"' tells a model to treat the 4,861 fraud rows as
# just as important as the 834,000 genuine ones. Without it, a model can get
# 99.4% accuracy by simply never predicting fraud.
fraud_count = int(y_train.sum())
genuine_count = int(len(y_train) - fraud_count)
imbalance_ratio = genuine_count / fraud_count

print()
print("Imbalance ratio: 1 fraud for every %.0f genuine transactions" % imbalance_ratio)

models_to_try = [
    {
        "name": "Logistic Regression",
        "model": LogisticRegression(class_weight="balanced", max_iter=1000,
                                    random_state=RANDOM_SEED),
        "needs_scaling": True,
        "sample_size": None,
    },
    {
        "name": "Decision Tree",
        "model": DecisionTreeClassifier(class_weight="balanced", max_depth=12,
                                        min_samples_leaf=50, random_state=RANDOM_SEED),
        "needs_scaling": False,
        "sample_size": None,
    },
    {
        "name": "Random Forest",
        "model": RandomForestClassifier(n_estimators=120, max_depth=18,
                                        min_samples_leaf=5, class_weight="balanced",
                                        n_jobs=-1, random_state=RANDOM_SEED),
        "needs_scaling": False,
        "sample_size": None,
    },
    {
        "name": "XGBoost",
        "model": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=imbalance_ratio,
                               eval_metric="aucpr", n_jobs=-1,
                               random_state=RANDOM_SEED),
        "needs_scaling": False,
        "sample_size": None,
    },
    {
        "name": "Support Vector Machine",
        "model": SVC(kernel="rbf", class_weight="balanced", C=1.0,
                     random_state=RANDOM_SEED),
        "needs_scaling": True,
        "sample_size": SVM_SAMPLE_SIZE,
    },
    {
        "name": "Neural Network (MLP)",
        "model": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=30,
                               early_stopping=True, random_state=RANDOM_SEED),
        "needs_scaling": True,
        "sample_size": MLP_SAMPLE_SIZE,
    },
]


# ===== STEP 4: Train each model and measure it =====
print()
print("--- Training models ---")
print("(the Random Forest and SVM take the longest)")

results = []
saved_models = {}
saved_scores = {}

for entry in models_to_try:
    name = entry["name"]
    model = entry["model"]

    print()
    print("Training:", name)

    # Pick the scaled or unscaled version of the data.
    if entry["needs_scaling"]:
        features_for_training = X_train_scaled
        features_for_testing = X_test_scaled
    else:
        features_for_training = X_train
        features_for_testing = X_test

    # Some models only train on a sample.
    if entry["sample_size"] is not None:
        features_for_training, target_for_training = make_sample(
            features_for_training, y_train, entry["sample_size"], RANDOM_SEED)
        print("   using a sample of %d rows" % len(features_for_training))
        note = "trained on %d row sample" % len(features_for_training)
    else:
        target_for_training = y_train
        note = "trained on all %d rows" % len(features_for_training)

    start_time = time.time()
    model.fit(features_for_training, target_for_training)
    seconds_taken = time.time() - start_time

    # Predict on the test set, which the model has never seen.
    predictions = model.predict(features_for_testing)
    scores = get_scores(model, features_for_testing)

    one_result = {}
    one_result["model"] = name
    one_result["accuracy"] = accuracy_score(y_test, predictions)
    one_result["precision"] = precision_score(y_test, predictions, zero_division=0)
    one_result["recall"] = recall_score(y_test, predictions, zero_division=0)
    one_result["f1"] = f1_score(y_test, predictions, zero_division=0)
    one_result["roc_auc"] = roc_auc_score(y_test, scores)
    one_result["pr_auc"] = average_precision_score(y_test, scores)
    one_result["training_seconds"] = round(seconds_taken, 1)
    one_result["note"] = note

    results.append(one_result)
    saved_models[name] = model
    saved_scores[name] = scores

    print("   done in %.1f seconds" % seconds_taken)
    print("   recall %.3f | precision %.3f | PR-AUC %.3f | ROC-AUC %.3f"
          % (one_result["recall"], one_result["precision"],
             one_result["pr_auc"], one_result["roc_auc"]))


# ===== STEP 5: An unsupervised model for comparison =====
# Isolation Forest never sees the answers. It just looks for odd-looking rows.
# We include it because in real life fraud labels arrive late or not at all.
print()
print("Training: Isolation Forest (unsupervised - it never sees the labels)")

isolation_model = IsolationForest(n_estimators=100, contamination=0.006,
                                  random_state=RANDOM_SEED, n_jobs=-1)
start_time = time.time()
isolation_model.fit(X_train_scaled)
seconds_taken = time.time() - start_time

# This model returns -1 for "odd" and 1 for "normal", so we convert to 0/1.
raw_predictions = isolation_model.predict(X_test_scaled)
isolation_predictions = np.where(raw_predictions == -1, 1, 0)

# A lower score means more unusual, so we flip the sign to line it up with the others.
isolation_scores = -isolation_model.score_samples(X_test_scaled)

isolation_result = {}
isolation_result["model"] = "Isolation Forest (unsupervised)"
isolation_result["accuracy"] = accuracy_score(y_test, isolation_predictions)
isolation_result["precision"] = precision_score(y_test, isolation_predictions, zero_division=0)
isolation_result["recall"] = recall_score(y_test, isolation_predictions, zero_division=0)
isolation_result["f1"] = f1_score(y_test, isolation_predictions, zero_division=0)
isolation_result["roc_auc"] = roc_auc_score(y_test, isolation_scores)
isolation_result["pr_auc"] = average_precision_score(y_test, isolation_scores)
isolation_result["training_seconds"] = round(seconds_taken, 1)
isolation_result["note"] = "no labels used"

results.append(isolation_result)
saved_scores["Isolation Forest (unsupervised)"] = isolation_scores

print("   recall %.3f | precision %.3f | PR-AUC %.3f"
      % (isolation_result["recall"], isolation_result["precision"], isolation_result["pr_auc"]))


# ===== STEP 6: Put the results in a table =====
print()
print("=" * 60)
print("MODEL COMPARISON (measured on the unseen test set)")
print("=" * 60)

comparison = pd.DataFrame(results)
comparison = comparison.sort_values("pr_auc", ascending=False)
comparison = comparison.reset_index(drop=True)

# Round the numbers so the table is readable.
for column in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
    comparison[column] = comparison[column].round(4)

print(comparison[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))

comparison.to_csv(os.path.join(TABLES_FOLDER, "model_comparison.csv"), index=False)

# The baseline is what you would score by guessing at random.
random_baseline = y_test.mean()
print()
print("For reference, a random guesser would score PR-AUC = %.4f" % random_baseline)
print("and a 'never fraud' model would score %.4f accuracy but 0 recall."
      % (1 - random_baseline))


# ===== Choosing which model to actually deploy =====
# The top of the table is often a photo finish. When two models are separated by
# less than PR_AUC_TIE_MARGIN we treat them as equally good, because a gap that
# small would move around if we simply reran with a different random seed.
#
# To break a tie we prefer a tree-based model, for a practical reason rather than
# a mathematical one: Step 5 has to explain individual decisions with SHAP, and
# SHAP is exact and fast on trees but slow and only approximate on a neural
# network. A bank that cannot explain why it declined a payment cannot use the
# model, however good its score is.
top_score = comparison["pr_auc"].iloc[0]
top_name = comparison["model"].iloc[0]

close_enough = comparison[comparison["pr_auc"] >= top_score - PR_AUC_TIE_MARGIN]
explainable_names = ["Random Forest", "XGBoost", "Decision Tree"]

best_model_name = top_name
for candidate_name in close_enough["model"]:
    if candidate_name in explainable_names:
        best_model_name = candidate_name
        break

print()
print("Highest PR-AUC          :", top_name, "(%.4f)" % top_score)

if len(close_enough) > 1:
    print("Models within %.2f of it :" % PR_AUC_TIE_MARGIN,
          ", ".join(close_enough["model"].tolist()))

if best_model_name != top_name:
    chosen_score = float(comparison[comparison["model"] == best_model_name]["pr_auc"].iloc[0])
    print()
    print("CHOSEN MODEL:", best_model_name, "(%.4f)" % chosen_score)
    print("Reason: it scores within %.4f of the best, which is inside the noise"
          % (top_score - chosen_score))
    print("margin, and unlike the top scorer it can be explained exactly with SHAP.")
    print("Being able to justify a declined payment matters more than 0.0008 of PR-AUC.")
else:
    print()
    print("CHOSEN MODEL:", best_model_name)


# ===== STEP 7: Compare ways of handling the imbalance =====
# The brief asks us to address imbalance, so we test three approaches
# on the same model to see which works best.
print()
print("--- Comparing three ways to handle the imbalance ---")
print("(using Logistic Regression on a 200,000 row sample so this stays quick)")

sample_features, sample_target = make_sample(X_train_scaled, y_train, 200000, RANDOM_SEED)
imbalance_results = []

# Approach 1: do nothing.
plain_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
plain_model.fit(sample_features, sample_target)
plain_scores = plain_model.predict_proba(X_test_scaled)[:, 1]
imbalance_results.append({
    "approach": "Do nothing",
    "recall": recall_score(y_test, plain_model.predict(X_test_scaled), zero_division=0),
    "precision": precision_score(y_test, plain_model.predict(X_test_scaled), zero_division=0),
    "pr_auc": average_precision_score(y_test, plain_scores),
})

# Approach 2: tell the model the classes matter equally.
weighted_model = LogisticRegression(class_weight="balanced", max_iter=1000,
                                    random_state=RANDOM_SEED)
weighted_model.fit(sample_features, sample_target)
weighted_scores = weighted_model.predict_proba(X_test_scaled)[:, 1]
imbalance_results.append({
    "approach": "Class weights",
    "recall": recall_score(y_test, weighted_model.predict(X_test_scaled), zero_division=0),
    "precision": precision_score(y_test, weighted_model.predict(X_test_scaled), zero_division=0),
    "pr_auc": average_precision_score(y_test, weighted_scores),
})

# Approach 3: SMOTE invents extra fraud examples so the classes are even.
smote = SMOTE(random_state=RANDOM_SEED)
smote_features, smote_target = smote.fit_resample(sample_features, sample_target)
print("SMOTE grew the sample from %d rows to %d rows" % (len(sample_features), len(smote_features)))

smote_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
smote_model.fit(smote_features, smote_target)
smote_scores = smote_model.predict_proba(X_test_scaled)[:, 1]
imbalance_results.append({
    "approach": "SMOTE oversampling",
    "recall": recall_score(y_test, smote_model.predict(X_test_scaled), zero_division=0),
    "precision": precision_score(y_test, smote_model.predict(X_test_scaled), zero_division=0),
    "pr_auc": average_precision_score(y_test, smote_scores),
})

imbalance_table = pd.DataFrame(imbalance_results)
for column in ["recall", "precision", "pr_auc"]:
    imbalance_table[column] = imbalance_table[column].round(4)

print()
print(imbalance_table.to_string(index=False))
imbalance_table.to_csv(os.path.join(TABLES_FOLDER, "imbalance_comparison.csv"), index=False)


# ===== STEP 8: Charts =====
print()
print("--- Drawing model charts ---")

# Figure 10: bar chart comparing the models
plt.figure(figsize=(11, 6))
chart_data = comparison.head(7)
x_positions = np.arange(len(chart_data))
width = 0.25

plt.bar(x_positions - width, chart_data["recall"], width, label="Recall", color="#C44E52")
plt.bar(x_positions, chart_data["precision"], width, label="Precision", color="#4C72B0")
plt.bar(x_positions + width, chart_data["pr_auc"], width, label="PR-AUC", color="#55A868")

plt.xticks(x_positions, chart_data["model"], rotation=20, ha="right")
plt.ylabel("Score (higher is better)")
plt.title("Model comparison on the unseen test set")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig10_model_comparison.png"), dpi=150)
plt.close()
print("  saved fig10_model_comparison.png")

# Figure 11: ROC curves
plt.figure(figsize=(8, 7))
for name in saved_scores:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, saved_scores[name])
    area = roc_auc_score(y_test, saved_scores[name])
    plt.plot(false_positive_rate, true_positive_rate, label="%s (%.3f)" % (name, area))

plt.plot([0, 1], [0, 1], "k--", label="Random guess (0.500)")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig11_roc_curves.png"), dpi=150)
plt.close()
print("  saved fig11_roc_curves.png")

# Figure 12: Precision-Recall curves. These matter more than ROC when
# the positive class is rare.
plt.figure(figsize=(8, 7))
for name in saved_scores:
    precision_values, recall_values, _ = precision_recall_curve(y_test, saved_scores[name])
    area = average_precision_score(y_test, saved_scores[name])
    plt.plot(recall_values, precision_values, label="%s (%.3f)" % (name, area))

plt.axhline(y=random_baseline, color="k", linestyle="--",
            label="Random guess (%.4f)" % random_baseline)
plt.xlabel("Recall (share of fraud we catch)")
plt.ylabel("Precision (share of alerts that are real)")
plt.title("Precision-Recall curves\nThis is the honest view when fraud is only 0.57% of the data")
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig12_pr_curves.png"), dpi=150)
plt.close()
print("  saved fig12_pr_curves.png")

# Figure 13: confusion matrix of the best model
best_scores = saved_scores[best_model_name]
best_predictions = (best_scores >= 0.5).astype(int)
if best_model_name == "Isolation Forest (unsupervised)":
    best_predictions = isolation_predictions

matrix = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(7, 6))
sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues",
            xticklabels=["Predicted genuine", "Predicted fraud"],
            yticklabels=["Actually genuine", "Actually fraud"])
plt.title("Confusion matrix - %s" % best_model_name)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig13_confusion_matrix.png"), dpi=150)
plt.close()
print("  saved fig13_confusion_matrix.png")


# ===== STEP 9: Choose the alert threshold using money =====
# By default a model flags fraud when the probability passes 0.5, but that
# number is arbitrary. The bank cares about pounds, not probabilities, so we
# test many thresholds and pick the one that saves the most money.
print()
print("--- Choosing the best alert threshold ---")

test_amounts = test["amt"].values
threshold_rows = []

# We start at 0.01 rather than 0.05. An earlier version of this script started at
# 0.05 and the "best" answer came out as exactly 0.05 - the very edge of the
# range, which is a warning sign that the real peak lies outside what was tested.
candidate_thresholds = np.concatenate([
    np.arange(0.01, 0.10, 0.01),
    np.arange(0.10, 1.00, 0.05),
])

for threshold in candidate_thresholds:
    flagged = (best_scores >= threshold).astype(int)

    caught_fraud = (flagged == 1) & (y_test == 1)
    missed_fraud = (flagged == 0) & (y_test == 1)
    false_alarms = (flagged == 1) & (y_test == 0)

    money_saved = test_amounts[caught_fraud].sum()
    review_cost = flagged.sum() * COST_OF_REVIEWING_ONE_ALERT
    upset_cost = false_alarms.sum() * COST_OF_UPSETTING_A_GOOD_CUSTOMER

    alerts = int(flagged.sum())
    precision_here = caught_fraud.sum() / alerts if alerts > 0 else 0.0

    threshold_rows.append({
        "threshold": round(float(threshold), 2),
        "alerts": alerts,
        "fraud_caught": int(caught_fraud.sum()),
        "fraud_missed": int(missed_fraud.sum()),
        "false_alarms": int(false_alarms.sum()),
        "recall": round(caught_fraud.sum() / y_test.sum(), 4),
        "precision": round(float(precision_here), 4),
        "money_saved": round(float(money_saved), 2),
        "review_cost": round(float(review_cost), 2),
        "customer_cost": round(float(upset_cost), 2),
        "net_benefit": round(float(money_saved - review_cost - upset_cost), 2),
    })

threshold_table = pd.DataFrame(threshold_rows)
threshold_table.to_csv(os.path.join(TABLES_FOLDER, "threshold_analysis.csv"), index=False)

best_row = threshold_table.loc[threshold_table["net_benefit"].idxmax()]
best_threshold = float(best_row["threshold"])

print(threshold_table[["threshold", "alerts", "fraud_caught", "fraud_missed",
                       "recall", "precision", "net_benefit"]].to_string(index=False))
print()
print("Best threshold for the money: %.2f" % best_threshold)
print("At that setting we catch %d of %d frauds (%.1f%% of them), raise %d alerts"
      % (best_row["fraud_caught"], int(y_test.sum()), best_row["recall"] * 100,
         best_row["alerts"]))
print("of which %.0f%% are genuine fraud, and net $%s."
      % (best_row["precision"] * 100, "{:,.0f}".format(best_row["net_benefit"])))

# Warn if the winner sits at either end of the range we searched.
if best_threshold <= candidate_thresholds[0] or best_threshold >= candidate_thresholds[-1]:
    print()
    print("WARNING: the best value is at the edge of the range we tested, so the")
    print("true best may lie outside it. Widen the range before trusting this.")


# ===== STEP 9b: Does the answer depend on our cost assumptions? =====
# The threshold we pick is only as trustworthy as the prices we put on things.
# So we redo the calculation with different assumptions about how much upsetting
# a genuine customer really costs, and see whether the answer moves.
print()
print("--- How sensitive is that to what we assume a false alarm costs? ---")

sensitivity_rows = []

for assumed_cost in [0.0, 15.0, 45.0, 100.0, 250.0]:
    best_value = None
    best_at = None

    for threshold in candidate_thresholds:
        flagged = (best_scores >= threshold).astype(int)
        caught_fraud = (flagged == 1) & (y_test == 1)
        false_alarms = (flagged == 1) & (y_test == 0)

        value = (test_amounts[caught_fraud].sum()
                 - flagged.sum() * COST_OF_REVIEWING_ONE_ALERT
                 - false_alarms.sum() * assumed_cost)

        if best_value is None or value > best_value:
            best_value = value
            best_at = threshold

    flagged = (best_scores >= best_at).astype(int)
    caught = int(((flagged == 1) & (y_test == 1)).sum())

    sensitivity_rows.append({
        "assumed_cost_per_false_alarm": assumed_cost,
        "best_threshold": round(float(best_at), 2),
        "alerts": int(flagged.sum()),
        "fraud_caught": caught,
        "recall": round(caught / int(y_test.sum()), 4),
        "net_benefit": round(float(best_value), 2),
    })

sensitivity_table = pd.DataFrame(sensitivity_rows)
print(sensitivity_table.to_string(index=False))
sensitivity_table.to_csv(os.path.join(TABLES_FOLDER, "cost_sensitivity.csv"), index=False)

print()
print("The threshold moves as the assumed cost changes, which is the honest")
print("finding: this is a business judgement about how much a wrongly blocked")
print("customer is worth, not something the data can decide on its own.")


# ===== STEP 9c: What if the team can only review so many alerts? =====
# Most fraud teams are staffed for a fixed workload. The cheapest threshold on
# paper is useless if it buries the team, so we also report the best threshold
# that keeps the alert count within what the team can actually handle.
print()
print("--- The option a real fraud team could actually staff ---")

affordable = threshold_table[threshold_table["alerts"] <= ALERTS_THE_TEAM_CAN_HANDLE]

if len(affordable) > 0:
    capacity_row = affordable.loc[affordable["net_benefit"].idxmax()]
    print("If the team can review at most %d alerts in this period, the best"
          % ALERTS_THE_TEAM_CAN_HANDLE)
    print("threshold is %.2f: %d alerts, %d frauds caught (%.1f%%), precision %.0f%%, net $%s."
          % (capacity_row["threshold"], capacity_row["alerts"],
             capacity_row["fraud_caught"], capacity_row["recall"] * 100,
             capacity_row["precision"] * 100,
             "{:,.0f}".format(capacity_row["net_benefit"])))
    capacity_option = {
        "max_alerts": ALERTS_THE_TEAM_CAN_HANDLE,
        "threshold": float(capacity_row["threshold"]),
        "alerts": int(capacity_row["alerts"]),
        "fraud_caught": int(capacity_row["fraud_caught"]),
        "recall": float(capacity_row["recall"]),
        "precision": float(capacity_row["precision"]),
        "net_benefit": float(capacity_row["net_benefit"]),
    }
else:
    capacity_option = None
    print("No threshold keeps the alert count within the team's capacity.")

# Figure 14: threshold vs money, with the recall and precision trade-off underneath
figure, (top_chart, bottom_chart) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

top_chart.plot(threshold_table["threshold"], threshold_table["net_benefit"],
               marker="o", color="#55A868", label="Net benefit ($)")
top_chart.axvline(x=best_threshold, color="#C44E52", linestyle="--",
                  label="Best threshold (%.2f)" % best_threshold)
top_chart.axvline(x=0.5, color="grey", linestyle=":", label="Default threshold (0.50)")
top_chart.set_ylabel("Net benefit in dollars")
top_chart.set_title("Picking the threshold by money, not by default\n"
                    "Fraud blocked, minus review time, minus the cost of upsetting good customers")
top_chart.legend()

bottom_chart.plot(threshold_table["threshold"], threshold_table["recall"],
                  marker="o", color="#C44E52", label="Recall (fraud we catch)")
bottom_chart.plot(threshold_table["threshold"], threshold_table["precision"],
                  marker="s", color="#4C72B0", label="Precision (alerts that are real)")
bottom_chart.axvline(x=best_threshold, color="#C44E52", linestyle="--")
bottom_chart.set_xlabel("Alert threshold")
bottom_chart.set_ylabel("Score")
bottom_chart.set_title("The trade-off behind that choice")
bottom_chart.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig14_threshold_tuning.png"), dpi=150)
plt.close()
print("  saved fig14_threshold_tuning.png")

# Figure 15b: how the best threshold shifts with the cost assumption
plt.figure(figsize=(9, 5))
plt.plot(sensitivity_table["assumed_cost_per_false_alarm"],
         sensitivity_table["best_threshold"], marker="o", color="#4C72B0", linewidth=2)
plt.xlabel("Assumed cost of upsetting one genuine customer ($)")
plt.ylabel("Threshold that makes the most money")
plt.title("The 'best' threshold depends on what we assume a false alarm costs\n"
          "This is a business judgement, not a result the data can settle")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig33_cost_sensitivity.png"), dpi=150)
plt.close()
print("  saved fig33_cost_sensitivity.png")


# ===== STEP 10: Save the best model and the results =====
print()
print("--- Saving ---")

if best_model_name in saved_models:
    joblib.dump(saved_models[best_model_name], os.path.join(MODELS_FOLDER, "best_model.joblib"))
    print("Saved best model:", best_model_name)

joblib.dump(isolation_model, os.path.join(MODELS_FOLDER, "isolation_forest.joblib"))

# Save the test-set scores so the bias audit can reuse them
# without having to train everything again.
score_frame = pd.DataFrame()
score_frame["fraud_score"] = best_scores
score_frame.to_csv(os.path.join(PROCESSED_FOLDER, "test_scores.csv"), index=False)

summary = {}
summary["best_model"] = best_model_name
summary["best_threshold"] = best_threshold
summary["random_baseline_pr_auc"] = round(float(random_baseline), 5)
summary["n_train"] = int(len(X_train))
summary["n_test"] = int(len(X_test))
summary["n_features"] = len(feature_columns)
summary["imbalance_ratio"] = round(float(imbalance_ratio), 1)
summary["cost_per_review"] = COST_OF_REVIEWING_ONE_ALERT
summary["cost_per_upset_customer"] = COST_OF_UPSETTING_A_GOOD_CUSTOMER
summary["highest_pr_auc_model"] = top_name
summary["highest_pr_auc"] = float(top_score)
summary["chosen_for_explainability"] = bool(best_model_name != top_name)
summary["cost_sensitivity"] = sensitivity_rows
summary["capacity_option"] = capacity_option

best_result = comparison[comparison["model"] == best_model_name].iloc[0]
summary["best_recall"] = float(best_result["recall"])
summary["best_precision"] = float(best_result["precision"])
summary["best_pr_auc"] = float(best_result["pr_auc"])
summary["best_roc_auc"] = float(best_result["roc_auc"])
summary["best_f1"] = float(best_result["f1"])
summary["best_accuracy"] = float(best_result["accuracy"])

summary["at_best_threshold"] = {
    "threshold": best_threshold,
    "alerts": int(best_row["alerts"]),
    "fraud_caught": int(best_row["fraud_caught"]),
    "fraud_missed": int(best_row["fraud_missed"]),
    "false_alarms": int(best_row["false_alarms"]),
    "recall": float(best_row["recall"]),
    "precision": float(best_row["precision"]),
    "money_saved": float(best_row["money_saved"]),
    "review_cost": float(best_row["review_cost"]),
    "customer_cost": float(best_row["customer_cost"]),
    "net_benefit": float(best_row["net_benefit"]),
}

summary["total_fraud_value_in_test"] = round(float(test_amounts[y_test == 1].sum()), 2)

json_path = os.path.join(TABLES_FOLDER, "model_results.json")
json_file = open(json_path, "w")
json.dump(summary, json_file, indent=2)
json_file.close()

print("Saved results to:", json_path)
print()
print("Done. Next step: python src/06_clustering.py")
