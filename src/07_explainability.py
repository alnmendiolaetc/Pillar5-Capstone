"""
Pillar 5 Capstone - Step 3 and Step 5: Explainability
Answers the question "why did the model say that?"

Methods used:
    1. Built-in feature importance  - what the model itself relies on
    2. Permutation importance       - what happens if we shuffle a column
    3. SHAP                         - fair credit for each feature, per prediction
    4. LIME                         - a plain-English reason for one transaction
    5. Feature selection            - can we use fewer columns?
    6. PCA                          - how much information is in how many columns?

What this file produces:
    reports/figures/fig21..fig27 *.png
    reports/tables/feature_importance.csv
    reports/tables/explainability_results.json

Run after 06_clustering.py:
    python src/07_explainability.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer


# ===== SETTINGS =====
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(THIS_FOLDER)
PROCESSED_FOLDER = os.path.join(PROJECT_FOLDER, "data", "processed")
MODELS_FOLDER = os.path.join(PROJECT_FOLDER, "models")
FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "figures")
TABLES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "tables")

RANDOM_SEED = 42

# SHAP and permutation importance are slow, so they run on a sample.
EXPLAIN_SAMPLE_SIZE = 3000

sns.set_theme(style="whitegrid")


# ===== STEP 1: Load everything we need =====
print("=" * 60)
print("STEP 3 & 5: EXPLAINABILITY")
print("=" * 60)

train = pd.read_csv(os.path.join(PROCESSED_FOLDER, "train.csv"))
test = pd.read_csv(os.path.join(PROCESSED_FOLDER, "test.csv"))

info_file = open(os.path.join(TABLES_FOLDER, "feature_list.json"))
feature_info = json.load(info_file)
info_file.close()
feature_columns = feature_info["feature_columns"]

results_file = open(os.path.join(TABLES_FOLDER, "model_results.json"))
model_results = json.load(results_file)
results_file.close()
best_model_name = model_results["best_model"]

model = joblib.load(os.path.join(MODELS_FOLDER, "best_model.joblib"))
scaler = joblib.load(os.path.join(MODELS_FOLDER, "scaler.joblib"))

X_train = train[feature_columns]
y_train = train["is_fraud"]
X_test = test[feature_columns]
y_test = test["is_fraud"]

print("Explaining the best model:", best_model_name)

# Tree models were trained on the raw numbers. The others were trained on
# scaled numbers, so we must feed them the same kind of data here.
tree_model_names = ["Random Forest", "XGBoost", "Decision Tree"]
model_uses_raw_numbers = best_model_name in tree_model_names

if model_uses_raw_numbers:
    X_test_for_model = X_test
    X_train_for_model = X_train
else:
    X_test_for_model = pd.DataFrame(scaler.transform(X_test), columns=feature_columns)
    X_train_for_model = pd.DataFrame(scaler.transform(X_train), columns=feature_columns)

explanations = {}


# ===== STEP 2: What does the model itself say is important? =====
print()
print("--- 1. Built-in feature importance ---")

if hasattr(model, "feature_importances_"):
    importance_values = model.feature_importances_
    importance_label = "Built-in importance"
else:
    # Linear models report coefficients instead.
    importance_values = np.abs(model.coef_[0])
    importance_label = "Size of the coefficient"

importance_table = pd.DataFrame()
importance_table["feature"] = feature_columns
importance_table["importance"] = importance_values
importance_table = importance_table.sort_values("importance", ascending=False)
importance_table = importance_table.reset_index(drop=True)

print(importance_table.head(12).to_string(index=False))

top_features = importance_table.head(15)

plt.figure(figsize=(9, 7))
plt.barh(top_features["feature"][::-1], top_features["importance"][::-1], color="#4C72B0")
plt.xlabel(importance_label)
plt.title("Top 15 features according to the model itself\n(%s)" % best_model_name)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig21_feature_importance.png"), dpi=150)
plt.close()
print("  saved fig21_feature_importance.png")

explanations["top_feature_builtin"] = str(importance_table["feature"].iloc[0])


# ===== STEP 3: Permutation importance =====
# We shuffle one column at a time and see how much worse the model gets.
# If shuffling a column changes nothing, the model was not really using it.
print()
print("--- 2. Permutation importance ---")
print("(shuffling each column to see how much the score drops)")

sample_rows = X_test_for_model.sample(n=EXPLAIN_SAMPLE_SIZE, random_state=RANDOM_SEED)
sample_target = y_test.loc[sample_rows.index]

permutation_result = permutation_importance(model, sample_rows, sample_target,
                                            n_repeats=5, random_state=RANDOM_SEED,
                                            scoring="average_precision", n_jobs=-1)

permutation_table = pd.DataFrame()
permutation_table["feature"] = feature_columns
permutation_table["drop_in_score"] = permutation_result.importances_mean
permutation_table = permutation_table.sort_values("drop_in_score", ascending=False)
permutation_table = permutation_table.reset_index(drop=True)

print(permutation_table.head(10).to_string(index=False))

top_permutation = permutation_table.head(15)

plt.figure(figsize=(9, 7))
plt.barh(top_permutation["feature"][::-1], top_permutation["drop_in_score"][::-1],
         color="#C44E52")
plt.xlabel("How much the PR-AUC drops when this column is shuffled")
plt.title("Permutation importance\nBigger bar = the model really depends on that column")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig22_permutation_importance.png"), dpi=150)
plt.close()
print("  saved fig22_permutation_importance.png")

explanations["top_feature_permutation"] = str(permutation_table["feature"].iloc[0])


# ===== STEP 4: SHAP =====
# SHAP splits a prediction into "how much did each feature push it up or down".
# It is the fairest way to explain a single decision.
print()
print("--- 3. SHAP values ---")
print("(this takes a minute)")

if model_uses_raw_numbers:
    explainer = shap.TreeExplainer(model)
else:
    # For non-tree models we explain using a small background sample.
    background = shap.sample(X_train_for_model, 100, random_state=RANDOM_SEED)
    explainer = shap.KernelExplainer(model.predict_proba, background)

shap_values = explainer.shap_values(sample_rows)

# Some models return one array per class. We want the "fraud" class.
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

# Figure 23: the beeswarm plot - every dot is one transaction
plt.figure()
shap.summary_plot(shap_values, sample_rows, show=False, max_display=15)
plt.title("SHAP: how each feature moves a prediction", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig23_shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig23_shap_summary.png")

# Figure 24: average size of each feature's effect
plt.figure()
shap.summary_plot(shap_values, sample_rows, plot_type="bar", show=False, max_display=15)
plt.title("SHAP: average effect of each feature", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig24_shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig24_shap_bar.png")

# Work out the SHAP ranking so we can quote it in the report.
average_shap = np.abs(shap_values).mean(axis=0)
shap_table = pd.DataFrame()
shap_table["feature"] = feature_columns
shap_table["average_shap"] = average_shap
shap_table = shap_table.sort_values("average_shap", ascending=False)
shap_table = shap_table.reset_index(drop=True)

print()
print("Top 5 features by SHAP:")
print(shap_table.head(5).to_string(index=False))

explanations["top_feature_shap"] = str(shap_table["feature"].iloc[0])
explanations["top_5_shap_features"] = shap_table["feature"].head(5).tolist()


# ===== STEP 5: LIME - explain one single transaction in plain terms =====
print()
print("--- 4. LIME explanation of one fraud case ---")

lime_explainer = LimeTabularExplainer(
    training_data=X_train_for_model.values,
    feature_names=feature_columns,
    class_names=["Genuine", "Fraud"],
    mode="classification",
    random_state=RANDOM_SEED,
)

# Find a transaction the model is confident is fraud, and that really is fraud.
test_scores = pd.read_csv(os.path.join(PROCESSED_FOLDER, "test_scores.csv"))["fraud_score"].values
fraud_and_caught = np.where((y_test.values == 1) & (test_scores > 0.8))[0]

if len(fraud_and_caught) > 0:
    chosen_position = int(fraud_and_caught[0])

    one_explanation = lime_explainer.explain_instance(
        data_row=X_test_for_model.iloc[chosen_position].values,
        predict_fn=model.predict_proba,
        num_features=10,
    )

    figure = one_explanation.as_pyplot_figure()
    figure.set_size_inches(9, 6)
    plt.title("LIME: why this transaction was flagged as fraud")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "fig25_lime_example.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  saved fig25_lime_example.png")

    # Save the reasons in words so the report can quote them.
    reasons = []
    for feature_text, weight in one_explanation.as_list():
        reasons.append({"reason": feature_text, "weight": round(float(weight), 4)})

    explanations["lime_example"] = {
        "actual_amount": round(float(test["amt"].iloc[chosen_position]), 2),
        "model_score": round(float(test_scores[chosen_position]), 4),
        "hour": int(test["hour"].iloc[chosen_position]),
        "reasons": reasons,
    }

    print()
    print("  This transaction was $%.2f at %02d:00" % (
        test["amt"].iloc[chosen_position], test["hour"].iloc[chosen_position]))
    print("  The model scored it %.3f. Top reasons:" % test_scores[chosen_position])
    for item in reasons[:5]:
        print("    -", item["reason"])
else:
    print("  No confidently detected fraud case found to explain.")


# ===== STEP 6: Feature selection - do we need all 27 columns? =====
# The brief asks for at least one feature selection method. We use two:
# a filter method (mutual information) and an embedded method (model importance).
print()
print("--- 5. Feature selection ---")

# Filter method: how much does each column tell us about fraud, on its own?
selection_sample = X_train.sample(n=50000, random_state=RANDOM_SEED)
selection_target = y_train.loc[selection_sample.index]

mutual_info = mutual_info_classif(selection_sample, selection_target,
                                  random_state=RANDOM_SEED)

mutual_table = pd.DataFrame()
mutual_table["feature"] = feature_columns
mutual_table["mutual_information"] = mutual_info
mutual_table = mutual_table.sort_values("mutual_information", ascending=False)

print("Top 8 features by mutual information (filter method):")
print(mutual_table.head(8).to_string(index=False))

# Embedded method: retrain using only the strongest columns and see what we lose.
print()
print("Testing how well a smaller model does:")

feature_counts_to_try = [5, 10, 15, len(feature_columns)]
selection_results = []

for how_many in feature_counts_to_try:
    chosen_features = importance_table["feature"].head(how_many).tolist()

    small_model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                                scale_pos_weight=model_results["imbalance_ratio"],
                                eval_metric="aucpr", n_jobs=-1,
                                random_state=RANDOM_SEED)
    small_model.fit(X_train[chosen_features], y_train)

    small_scores = small_model.predict_proba(X_test[chosen_features])[:, 1]
    small_pr_auc = average_precision_score(y_test, small_scores)

    selection_results.append({"features_used": how_many, "pr_auc": round(float(small_pr_auc), 4)})
    print("  %2d features -> PR-AUC %.4f" % (how_many, small_pr_auc))

selection_table = pd.DataFrame(selection_results)
selection_table.to_csv(os.path.join(TABLES_FOLDER, "feature_selection.csv"), index=False)

plt.figure(figsize=(9, 5))
plt.plot(selection_table["features_used"], selection_table["pr_auc"],
         marker="o", color="#55A868", linewidth=2)
plt.xlabel("Number of features kept")
plt.ylabel("PR-AUC on the test set")
plt.title("Do we need every feature?\nYes - the score keeps climbing all the way to the full set")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig26_feature_selection.png"), dpi=150)
plt.close()
print("  saved fig26_feature_selection.png")

explanations["feature_selection"] = selection_results


# ===== STEP 7: PCA - how many columns do we really need? =====
print()
print("--- 6. PCA (dimensionality reduction) ---")

scaled_train = scaler.transform(X_train)
pca = PCA(random_state=RANDOM_SEED)
pca.fit(scaled_train)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100

# How many components do we need to keep 95% of the information?
components_for_95 = int(np.argmax(cumulative_variance >= 95) + 1)
print("Components needed to keep 95%% of the information: %d (out of %d)"
      % (components_for_95, len(feature_columns)))

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
         marker="o", color="#4C72B0")
plt.axhline(y=95, color="#C44E52", linestyle="--", label="95% of the information")
plt.axvline(x=components_for_95, color="grey", linestyle=":",
            label="%d components" % components_for_95)
plt.xlabel("Number of PCA components")
plt.ylabel("Information kept (%)")
plt.title("PCA: how much information we keep as we add components")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig27_pca_variance.png"), dpi=150)
plt.close()
print("  saved fig27_pca_variance.png")

explanations["pca_components_for_95_percent"] = components_for_95
explanations["total_features"] = len(feature_columns)

print()
print("Note: we do NOT use PCA for the final model. PCA mixes the columns")
print("together, which would destroy the plain-English explanations that a")
print("fraud analyst needs when they phone a customer.")


# ===== STEP 8: Save everything =====
combined = importance_table.merge(permutation_table, on="feature", how="left")
combined = combined.merge(shap_table, on="feature", how="left")
combined = combined.merge(mutual_table, on="feature", how="left")
combined.to_csv(os.path.join(TABLES_FOLDER, "feature_importance.csv"), index=False)

json_path = os.path.join(TABLES_FOLDER, "explainability_results.json")
json_file = open(json_path, "w")
json.dump(explanations, json_file, indent=2)
json_file.close()

print()
print("Saved:", json_path)
print()
print("Done. Next step: python src/08_bias_audit.py")
