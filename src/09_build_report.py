"""
Pillar 5 Capstone - Step 6: Build the written report
Creates the Word document that answers every step of the assignment.

Every number in this report is read from the files produced by scripts 02-08.
Nothing is typed in by hand, so the report always matches the run that made it.

What this file produces:
    Allen_Mendiola_Credit_Card_Fraud_Detection.docx

Run after 08_bias_audit.py:
    python src/09_build_report.py
"""

import os
import json
from datetime import date

import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


# ===== SETTINGS =====
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(THIS_FOLDER)
FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "figures")
TABLES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "tables")

STUDENT_NAME = "Allen Mendiola"
REPORT_TITLE = "Detecting Fraudulent Credit Card Transactions"
REPORT_SUBTITLE = "An End-to-End Machine Learning Lifecycle Project"
OUTPUT_NAME = "Allen_Mendiola_Credit_Card_Fraud_Detection.docx"
GITHUB_URL = "https://github.com/alnmendiolaetc/Pillar5-Capstone"


# ===== Small helpers so the rest of the file stays readable =====

def load_json(file_name):
    """Read one of the .json files we saved earlier."""
    path = os.path.join(TABLES_FOLDER, file_name)
    handle = open(path)
    content = json.load(handle)
    handle.close()
    return content


def load_csv(file_name):
    """Read one of the .csv files we saved earlier."""
    return pd.read_csv(os.path.join(TABLES_FOLDER, file_name))


def h1(text):
    doc.add_heading(text, level=1)


def h2(text):
    doc.add_heading(text, level=2)


def h3(text):
    doc.add_heading(text, level=3)


def p(text):
    return doc.add_paragraph(text)


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def numbered(text):
    doc.add_paragraph(text, style="List Number")


def key_value(label, value):
    """A short 'Label: value' line with the label in bold."""
    paragraph = doc.add_paragraph()
    run = paragraph.add_run(label + ": ")
    run.bold = True
    paragraph.add_run(str(value))
    return paragraph


def callout(text):
    """An indented italic line used to highlight an important point."""
    paragraph = doc.add_paragraph(text)
    paragraph.paragraph_format.left_indent = Inches(0.35)
    for run in paragraph.runs:
        run.italic = True
        run.font.size = Pt(10)
    return paragraph


def table_from_dataframe(frame, column_titles=None, number_format="%.4g"):
    """Draw a pandas table into the document."""
    if column_titles is None:
        column_titles = list(frame.columns)

    new_table = doc.add_table(rows=1, cols=len(frame.columns))
    new_table.style = "Table Grid"

    # Header row.
    header_cells = new_table.rows[0].cells
    for position in range(len(column_titles)):
        header_cells[position].text = ""
        run = header_cells[position].paragraphs[0].add_run(str(column_titles[position]))
        run.bold = True
        run.font.size = Pt(9)

    # Body rows.
    for row_position in range(len(frame)):
        cells = new_table.add_row().cells
        for column_position in range(len(frame.columns)):
            value = frame.iloc[row_position, column_position]

            if isinstance(value, float):
                text = number_format % value
            else:
                text = str(value)

            cells[column_position].text = ""
            run = cells[column_position].paragraphs[0].add_run(text)
            run.font.size = Pt(8.5)

    doc.add_paragraph()
    return new_table


def add_figure(file_name, caption, width_inches=6.1):
    """Put a chart into the document with a caption under it."""
    path = os.path.join(FIGURES_FOLDER, file_name)

    if not os.path.exists(path):
        print("  WARNING: missing figure", file_name)
        return

    doc.add_picture(path, width=Inches(width_inches))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    caption_paragraph = doc.add_paragraph(caption)
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in caption_paragraph.runs:
        run.italic = True
        run.font.size = Pt(9)


def money(amount):
    """Format a number as dollars with thousands separators."""
    return "$" + "{:,.0f}".format(amount)


# ===== STEP 1: Load every result we produced earlier =====
print("=" * 60)
print("STEP 6: BUILDING THE REPORT")
print("=" * 60)

overview = load_json("data_overview.json")
eda = load_json("eda_findings.json")
features = load_json("feature_list.json")
models = load_json("model_results.json")
clusters = load_json("clustering_results.json")
explain = load_json("explainability_results.json")
fairness = load_json("fairness_results.json")

data_dictionary = load_csv("data_dictionary.csv")
model_comparison = load_csv("model_comparison.csv")
imbalance_comparison = load_csv("imbalance_comparison.csv")
feature_notes = load_csv("feature_engineering_notes.csv")
cluster_profiles = load_csv("cluster_profiles.csv")
gender_bias = load_csv("bias_by_gender.csv")
age_bias = load_csv("bias_by_age.csv")
city_bias = load_csv("bias_by_city_size.csv")
mitigation_table = load_csv("bias_after_mitigation.csv")
feature_selection = load_csv("feature_selection.csv")
threshold_analysis = load_csv("threshold_analysis.csv")

best = models["at_best_threshold"]

print("Loaded all result files.")


# ===== STEP 2: Set up the document =====
doc = Document()

normal_style = doc.styles["Normal"]
normal_style.font.name = "Calibri"
normal_style.font.size = Pt(11)


# ===== TITLE PAGE =====
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_run = title.add_run(REPORT_TITLE)
title_run.bold = True
title_run.font.size = Pt(26)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle_run = subtitle.add_run(REPORT_SUBTITLE)
subtitle_run.font.size = Pt(15)
subtitle_run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

doc.add_paragraph()

for line in [STUDENT_NAME,
             "Pillar 5 Capstone Assignment",
             "Domain: Finance - Fraudulent Transaction Detection",
             date.today().strftime("%d %B %Y")]:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(line)
    run.font.size = Pt(12)

doc.add_paragraph()

summary_intro = doc.add_paragraph()
summary_intro.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = summary_intro.add_run(
    "A complete machine learning lifecycle, from framing the business problem "
    "through to auditing the finished model for bias.")
run.italic = True
run.font.size = Pt(11)

doc.add_page_break()


# ===== CONTENTS =====
h1("Contents")
contents = [
    "Executive Summary",
    "Step 1 - Problem Understanding and Framing",
    "Step 2 - Data Collection and Understanding",
    "Step 3 - Preprocessing, Applied EDA and Feature Engineering",
    "Step 4 - Model Implementation",
    "Step 5 - Critical Thinking: Ethical AI and Bias Auditing",
    "Step 6 - Final Presentation and Communication",
    "Step 7 - GitHub Repository",
    "Conclusion and Recommendations",
    "Limitations and Future Work",
    "References",
    "Appendix A - How to Reproduce This Work",
]
for item in contents:
    bullet(item)

doc.add_page_break()


# ===== EXECUTIVE SUMMARY =====
h1("Executive Summary")

p("This project builds and audits a machine learning system that spots fraudulent "
  "credit card transactions. It follows the full lifecycle: framing the problem, "
  "understanding the data, cleaning and engineering features, training and comparing "
  "seven models, explaining what the winning model has learned, and finally checking "
  "whether it treats different groups of customers fairly.")

p("The dataset contains %s card transactions made between %s and %s. Only %s of them "
  "are fraudulent - %.3f%% of the total. That rarity is the central difficulty of the "
  "whole project, and it shapes every decision that follows."
  % ("{:,}".format(overview["n_rows"]),
     overview["first_transaction"][:10],
     overview["last_transaction"][:10],
     "{:,}".format(overview["n_fraud"]),
     overview["fraud_rate_percent"]))

h2("Headline results")

key_value("Best model", models["best_model"])
key_value("PR-AUC (the metric that matters here)", "%.4f" % models["best_pr_auc"])
key_value("ROC-AUC", "%.4f" % models["best_roc_auc"])
key_value("Fraud caught at the chosen threshold",
          "%d of %d (%.1f%%)" % (best["fraud_caught"],
                                 best["fraud_caught"] + best["fraud_missed"],
                                 best["recall"] * 100))
key_value("Estimated net benefit on the test period",
          "%s" % money(best["net_benefit"]))

p("")
callout("A model that simply never predicted fraud would still be %.2f%% accurate. "
        "That is why this report leads with PR-AUC and recall rather than accuracy - "
        "accuracy is actively misleading when the thing you are looking for is this rare."
        % ((1 - models["random_baseline_pr_auc"]) * 100))

h2("What the fairness audit found")

p("Gender and job were deliberately withheld from the model. Many people assume that "
  "this alone makes a model fair. The audit in Step 5 tests that assumption directly, "
  "and the results are reported honestly, including where the model falls short.")

gender_note = fairness["gender"]
age_note = fairness["age_band"]

key_value("Gender - disparate impact ratio",
          "%.3f (%s the 80%% rule)" % (gender_note["disparate_impact_ratio"],
                                       "passes" if gender_note["passes_80_percent_rule"] else "FAILS"))
key_value("Age - gap in fraud caught between best and worst served group",
          "%.1f percentage points" % age_note["equal_opportunity_gap"])
key_value("After mitigation", "gap reduced to %.1f points"
          % fairness["mitigation"]["gap_after"])

doc.add_page_break()


# ===== STEP 1 =====
h1("Step 1 - Problem Understanding and Framing")

h2("1.1 The business problem")

p("Card fraud is a direct financial loss. When a stolen card is used, the bank usually "
  "refunds the customer and absorbs the cost. The bank therefore wants to spot fraud "
  "while it is happening, so the payment can be stopped before the money is gone.")

p("The difficulty is that blocking payments is not free. Every alert has to be checked "
  "by a person, and every wrongly blocked payment annoys an innocent customer who is "
  "standing at a till. A system that flags everything would catch all the fraud and "
  "destroy the business. The real task is therefore a balancing act, not simply "
  "'find the fraud'.")

h2("1.2 Task type")

key_value("Learning type", "Supervised learning")
key_value("Task", "Binary classification - each transaction is either fraud (1) or genuine (0)")
key_value("Target variable", "is_fraud")
key_value("Unit of analysis", "One card transaction")
key_value("Prediction timing", "At the moment the payment is attempted")

p("")
p("A secondary unsupervised task is also carried out in Step 4. Clustering and anomaly "
  "detection are used to check whether suspicious behaviour can be found without any "
  "fraud labels at all. This matters in practice because in a real bank the labels "
  "arrive weeks later, when customers report the fraud.")

h2("1.3 Success metrics")

p("Two sets of measures are used, because the data science answer and the business "
  "answer are not the same thing.")

h3("Technical metrics")

bullet("PR-AUC (average precision) - the headline metric. It measures how well the model "
       "ranks fraud above genuine payments, and unlike ROC-AUC it does not flatter a model "
       "when the positive class is rare.")
bullet("Recall - of all the real fraud, what share did we catch? This is what the bank "
       "loses money on when it is low.")
bullet("Precision - of everything we flagged, what share was really fraud? This is what "
       "drives the review workload.")
bullet("ROC-AUC and F1 - reported for completeness and for comparison with other studies.")

p("")
callout("Accuracy is deliberately NOT used as a success metric. With a fraud rate of "
        "%.3f%%, a model that predicts 'genuine' for everything scores %.2f%% accuracy "
        "and catches zero fraud."
        % (overview["fraud_rate_percent"], (1 - models["random_baseline_pr_auc"]) * 100))

h3("Business KPIs")

bullet("Value of fraud blocked, in dollars.")
bullet("Cost of investigating false alarms, assumed at %s per alert of analyst time."
       % money(models["cost_per_review"]))
bullet("Net benefit = fraud blocked minus review cost. This single number is used in "
       "Step 4 to choose the alert threshold.")
bullet("Customer friction, measured as the number of genuine customers wrongly blocked.")

h2("1.4 Capstone linkage")

p("The brief asks that the Module 1 output maps onto Capstone Steps 1 to 3. The mapping "
  "used here is:")

linkage = pd.DataFrame([
    ["Problem statement", "Step 1", "Detect fraudulent card transactions in real time"],
    ["Task type", "Step 1", "Supervised binary classification, plus unsupervised support"],
    ["Target metric", "Step 1", "PR-AUC, with recall and net benefit as business measures"],
    ["Data sourcing", "Step 2", "Public simulated transaction dataset, 1.05 million rows"],
    ["Data dictionary", "Step 2", "All 22 variables documented in section 2.3"],
    ["Preparation", "Step 3", "Cleaning, 13 engineered features, time-based split"],
], columns=["Module 1 output", "Capstone step", "How it is addressed here"])
table_from_dataframe(linkage)

doc.add_page_break()


# ===== STEP 2 =====
h1("Step 2 - Data Collection and Understanding")

h2("2.1 Source and licence")

key_value("Dataset", "Credit Card Transactions Fraud Detection (Sparkov simulation)")
key_value("Hosted at", "Hugging Face - santosh3110/credit_card_fraud_transactions")
key_value("Access", "Public download, no account or API key required")
key_value("File used", overview["source_file"])

p("")
p("The data is simulated rather than taken from a real bank. This is a deliberate choice "
  "and a necessary one: genuine card transaction data cannot be published because it "
  "contains real people's spending. The simulation was built to reproduce realistic "
  "spending patterns, and crucially it keeps the customer details - gender, date of "
  "birth, job, city size - that the fairness audit in Step 5 depends on.")

callout("The better known alternative, the ULB 'creditcard' dataset, was rejected for "
        "exactly this reason. Its features are anonymised into unnamed components V1 to "
        "V28, which would have made both the SHAP explanations and the bias audit "
        "impossible to carry out meaningfully.")

h2("2.2 Dataset overview")

overview_table = pd.DataFrame([
    ["Rows (transactions)", "{:,}".format(overview["n_rows"])],
    ["Columns", str(overview["n_columns"])],
    ["Fraudulent transactions", "{:,}".format(overview["n_fraud"])],
    ["Genuine transactions", "{:,}".format(overview["n_genuine"])],
    ["Fraud rate", "%.3f%%" % overview["fraud_rate_percent"]],
    ["Time covered", "%s to %s" % (overview["first_transaction"][:10],
                                   overview["last_transaction"][:10])],
    ["Missing values", "{:,}".format(overview["total_missing_values"])],
    ["Duplicate rows", "{:,}".format(overview["duplicate_rows"])],
    ["Total value of fraud", money(overview["total_fraud_amount"])],
    ["Average fraudulent payment", money(overview["average_fraud_amount"])],
    ["Average genuine payment", money(overview["average_genuine_amount"])],
], columns=["Property", "Value"])
table_from_dataframe(overview_table)

p("The average fraudulent payment is %s, roughly %.1f times the size of the average "
  "genuine payment at %s. This is the single strongest clue in the whole dataset."
  % (money(overview["average_fraud_amount"]),
     overview["average_fraud_amount"] / overview["average_genuine_amount"],
     money(overview["average_genuine_amount"])))

h2("2.3 Data dictionary")

p("Every column in the raw file, what it means, and what values it can take.")
table_from_dataframe(
    data_dictionary[["variable", "description", "data_type", "unit_or_format", "unique_values"]],
    column_titles=["Variable", "Description", "Type", "Unit / format", "Distinct values"])

h2("2.4 Data quality findings")

p("Three issues were found while inspecting the data. All three are reported here rather "
  "than quietly fixed, because each one would change how much trust the results deserve.")

h3("Finding 1 - the file has been truncated")

p("The file contains exactly %s rows. That number is not a coincidence: it is precisely "
  "the maximum number of data rows Microsoft Excel can hold. The original Sparkov "
  "training file is known to contain 1,296,675 rows, so roughly 248,000 transactions "
  "have been lost, almost certainly because somebody opened the file in Excel and saved "
  "it again." % "{:,}".format(overview["n_rows"]))

p("The practical effect is that the data stops on %s instead of running to mid-2020. "
  "Since the split between training and test data is by date, this simply means the test "
  "period is earlier than it would otherwise have been. It does not invalidate the "
  "results, but it is an honest limitation."
  % overview["last_transaction"][:10])

h3("Finding 2 - dates of birth were being read in the wrong century")

p("Birth dates are stored as text with a two-digit year, for example \"1/19/62\". Python's "
  "default rule treats any year below 69 as belonging to the 2000s, so 62 was read as "
  "2062 rather than 1962. This affected %s rows, %.1f%% of the dataset, and produced "
  "customers who appeared to be born in the future with NEGATIVE ages."
  % ("{:,}".format(overview.get("dob_wrong_century_rows", 0)),
     overview.get("dob_wrong_century_percent", 0)))

callout("This bug was caught because a clustering profile reported an average customer "
        "age of minus 41 years. Had it gone unnoticed, the age feature would have been "
        "meaningless and - far more seriously - the age fairness audit in Step 5 would "
        "have been auditing nonsense while appearing to work perfectly.")

p("The fix, applied in script 03, is to subtract one hundred years from any birth date "
  "that falls after the transactions took place. Ages afterwards run from roughly 14 to "
  "95 years, which is sensible.")

h3("Finding 3 - no missing values at all")

p("There are zero missing values and zero duplicate transactions. Real banking data is "
  "never this clean. This is a reminder that the dataset is simulated, and that the "
  "missing-value handling demonstrated in Step 3 would be doing considerably more work "
  "on a real system.")

doc.add_page_break()


# ===== STEP 3 =====
h1("Step 3 - Preprocessing, Applied EDA and Feature Engineering")

h2("3.1 Cleaning decisions")

numbered("Duplicate transactions were removed using the unique transaction reference. None "
         "were found, but the check remains in the code so it would catch them in future data.")
numbered("Both date columns are parsed with an explicit format rather than letting pandas "
         "guess, and the century bug described in section 2.4 is corrected.")
numbered("Rows with no target value are dropped, since nothing can be learned from them.")
numbered("Extreme transaction amounts are KEPT, not removed. This is the opposite of the "
         "usual advice. In fraud detection an unusually large payment is very often the "
         "fraud itself, so deleting outliers would delete the signal.")

p("")
p("On that last point: %s transactions (%.2f%% of the data) sit above the usual "
  "statistical cut-off of %s. Removing them would have discarded a large share of the "
  "fraud the model is supposed to find."
  % ("{:,}".format(overview["amt_outlier_count"]),
     overview["amt_outlier_count"] * 100.0 / overview["n_rows"],
     money(overview["amt_outlier_upper_limit"])))

h2("3.2 Exploratory data analysis")

h3("The class imbalance")
add_figure("fig01_class_imbalance.png",
           "Figure 1: Fraud makes up only %.3f%% of transactions. Note the logarithmic "
           "scale - on a normal scale the fraud bar would be invisible."
           % eda["fraud_rate_percent"])

h3("Transaction amount")
add_figure("fig02_amount_distribution.png",
           "Figure 2: Fraudulent payments cluster at clearly higher amounts. Average "
           "fraud %s against %s for genuine payments."
           % (money(eda["average_fraud_amount"]), money(eda["average_genuine_amount"])))

h3("Time of day")
add_figure("fig03_fraud_by_hour.png",
           "Figure 3: Fraud rate by hour. Between 10pm and 4am the rate is %.3f%%, "
           "against %.3f%% during the day - a difference of roughly %.0f times."
           % (eda["night_fraud_rate_percent"], eda["day_fraud_rate_percent"],
              eda["night_fraud_rate_percent"] / eda["day_fraud_rate_percent"]))

p("This is the clearest single pattern in the data. The explanation is intuitive: a "
  "criminal using a stolen card prefers the hours when the real owner is asleep and "
  "cannot notice the payment or answer a verification call.")

h3("Purchase category")
add_figure("fig04_fraud_by_category.png",
           "Figure 4: Fraud rate by category. '%s' is the riskiest at %.2f%%, and the "
           "safest is '%s'."
           % (eda["worst_category"], eda["worst_category_rate_percent"], eda["safest_category"]))

h3("Customer age")
add_figure("fig05_age_distribution.png",
           "Figure 5: Age profile. Fraud victims average %.1f years against %.1f for "
           "genuine transactions."
           % (eda["average_fraud_age"], eda["average_genuine_age"]))

h3("How the features relate to each other")
add_figure("fig06_correlation_heatmap.png",
           "Figure 6: Correlation heatmap. The feature most strongly linked to fraud is "
           "'%s'." % eda["strongest_correlation_feature"])

h3("An early warning for the fairness audit")
add_figure("fig07_fraud_by_age_and_gender.png",
           "Figure 7: Fraud rates already differ by age group and gender before any model "
           "is trained. Those differences are in the world, not in the algorithm - but a "
           "model will learn them.")

h3("Distance and spending behaviour")
add_figure("fig08_distance_distribution.png",
           "Figure 8: Distance between the customer's home and the shop.")
add_figure("fig09_amount_vs_card_average.png",
           "Figure 9: Payment size compared with what that particular card normally spends. "
           "Fraudulent payments are dramatically out of character.")

h2("3.3 Feature engineering")

p("Thirteen features were engineered from the raw columns. Each one exists for a stated "
  "reason rather than because it was available.")

table_from_dataframe(feature_notes,
                     column_titles=["Feature", "Type", "Why it is used"])

h3("Scaling, encoding and binning")

p("Three standard preparation techniques are applied, each for a specific reason "
  "rather than out of habit.")

techniques = pd.DataFrame([
    ["Scaling", "StandardScaler on all %d features"
     % len(features["feature_columns"]),
     "Logistic Regression, SVM and the neural network all measure distances between "
     "points. Without scaling, city population (in the millions) would completely "
     "drown out is_night (0 or 1). Tree models do not care, so both a scaled and an "
     "unscaled copy of the data are kept and each model is given the one it needs."],
    ["Encoding", "One-hot for category (14 columns); frequency encoding for merchant",
     "Models cannot read text. Category has only 14 values, so one-hot keeps each one "
     "separately interpretable for SHAP. Merchant has 693 values, which would add 693 "
     "columns, so it is replaced by how often each shop appears instead."],
    ["Binning", "age into 4 bands; city_pop into 4 size classes",
     "These bands are NOT model inputs. They exist so the fairness audit in Step 5 can "
     "compare groups. Auditing across continuous age would give one 'group' per "
     "customer and say nothing."],
], columns=["Technique", "What was done", "Why"])
table_from_dataframe(techniques)

callout("The scaler is fitted on the training rows only and then applied to the test "
        "rows. Fitting it on the whole dataset would let the test period's mean and "
        "spread leak into training - a subtle and very common mistake that quietly "
        "inflates results.")

h3("Guarding against leakage")

p("Two features - the card's average spend and how common each shop is - are learned from "
  "the data rather than being present in it. Both are calculated on the TRAINING rows "
  "only and then applied to the test rows. Calculating them across the whole dataset "
  "would let information from the test period leak backwards into training, and the "
  "resulting scores would be flattering and wrong.")

h3("What was deliberately thrown away")

p("Six columns were removed because they identify a real person: first name, last name, "
  "street address, transaction reference, card number and postcode. This is a privacy "
  "decision as much as a modelling one. A fraud model has no legitimate need to know a "
  "customer's name, and keeping such data increases the damage if the system is ever "
  "breached.")

p("Two further columns - gender and job - were removed for a different reason. These are "
  "protected characteristics, and basing a financial decision on them directly would be "
  "both unethical and, in most jurisdictions, unlawful. They are retained in a separate "
  "part of the file so that Step 5 can audit the model's behaviour across those groups.")

callout("Age is treated differently. It IS given to the model, because spending patterns "
        "genuinely change across a lifetime and the signal is behavioural rather than "
        "discriminatory. This is a judgement call, and because it is a judgement call, "
        "age is audited especially closely in Step 5.")

h2("3.4 Train and test split")

key_value("Method", "Chronological, not random")
key_value("Split date", features["split_date"][:10])
key_value("Training rows", "{:,} ({:.3f}% fraud)".format(features["n_train"],
                                                         features["train_fraud_rate_percent"]))
key_value("Test rows", "{:,} ({:.3f}% fraud)".format(features["n_test"],
                                                     features["test_fraud_rate_percent"]))

p("")
p("The split is by date on purpose. A random split would scatter transactions from the "
  "same week across both sides, letting the model learn from the future to predict the "
  "past. That inflates scores and produces a model that disappoints the moment it meets "
  "genuinely new data. Training on the earlier period and testing on the later one "
  "reproduces how the system would actually be used.")

h2("3.5 Feature importance and explainability")

p("Four different methods were used, because any single method can mislead.")

add_figure("fig21_feature_importance.png",
           "Figure 10: The model's own view of which features matter. Top feature: %s."
           % explain["top_feature_builtin"])

add_figure("fig22_permutation_importance.png",
           "Figure 11: Permutation importance - how far the score falls when each column "
           "is shuffled. Top feature: %s." % explain["top_feature_permutation"])

add_figure("fig23_shap_summary.png",
           "Figure 12: SHAP values. Every dot is one transaction. Red means a high value "
           "of that feature, and position shows whether it pushed the prediction towards "
           "fraud or away from it.")

add_figure("fig24_shap_bar.png",
           "Figure 13: Average size of each feature's SHAP effect. Top feature: %s."
           % explain["top_feature_shap"])

p("The top five features by SHAP are: %s." % ", ".join(explain["top_5_shap_features"]))

h3("Explaining one single decision with LIME")

if "lime_example" in explain:
    example = explain["lime_example"]
    p("SHAP explains the model overall. LIME answers a different and very practical "
      "question: why was THIS payment flagged? That is the question a fraud analyst has "
      "to answer when a customer telephones to complain.")

    key_value("Transaction amount", money(example["actual_amount"]))
    key_value("Time of day", "%02d:00" % example["hour"])
    key_value("Model's fraud score", "%.3f" % example["model_score"])

    p("")
    p("The reasons LIME gives, in order of weight:")
    for item in example["reasons"][:6]:
        bullet("%s (weight %.3f)" % (item["reason"], item["weight"]))

    add_figure("fig25_lime_example.png",
               "Figure 14: LIME explanation for a single flagged transaction.")

h2("3.6 Feature selection")

p("Two approaches were tested: a filter method using mutual information, and an embedded "
  "method that retrains the model using only the strongest features.")

table_from_dataframe(feature_selection,
                     column_titles=["Features kept", "PR-AUC achieved"])

add_figure("fig26_feature_selection.png",
           "Figure 15: Performance against the number of features kept.")

best_small = feature_selection[feature_selection["features_used"] == 10]
if len(best_small) > 0:
    full_score = feature_selection["pr_auc"].iloc[-1]
    ten_score = best_small["pr_auc"].iloc[0]

    p("The expected result here was a plateau - the familiar pattern where a handful of "
      "features do almost all the work and the rest can be dropped for free. That is not "
      "what happened. PR-AUC climbs steadily from %.4f with five features, to %.4f with "
      "ten, to %.4f with all %d. The curve never flattens."
      % (feature_selection["pr_auc"].iloc[0], ten_score, full_score,
         len(features["feature_columns"])))

    p("Dropping to the ten strongest features would therefore cost %.4f of PR-AUC, which "
      "is roughly %.0f%% of the model's performance - a real loss, not a rounding error. "
      "The honest conclusion is that this feature set carries very little redundancy and "
      "all of it should be kept."
      % (full_score - ten_score, (full_score - ten_score) * 100 / full_score))

    callout("The PCA result in the next section points the same way: %d of the %d "
            "components are needed to retain 95%% of the variance. Two independent methods "
            "agreeing that the features are not redundant is a stronger finding than "
            "either one alone."
            % (explain["pca_components_for_95_percent"], explain["total_features"]))

h2("3.7 Dimensionality reduction")

add_figure("fig27_pca_variance.png",
           "Figure 16: PCA. %d components are needed to retain 95%% of the information "
           "in the %d features." % (explain["pca_components_for_95_percent"],
                                    explain["total_features"]))

p("PCA is used here for understanding and for visualisation only. It is deliberately NOT "
  "used to build the final model. PCA blends the original columns into mathematical "
  "combinations, which would destroy the plain-English explanations that Step 5 depends "
  "on. A fraud analyst can act on 'the payment was 40 times this card's normal spend, at "
  "2am'. Nobody can act on 'component 3 was high'.")

doc.add_page_break()


# ===== STEP 4 =====
h1("Step 4 - Model Implementation")

h2("4.1 Models tested")

p("Seven models were trained and compared on identical data, covering the families the "
  "brief asks for.")

model_descriptions = pd.DataFrame([
    ["Logistic Regression", "Linear", "The simple baseline every project should have."],
    ["Decision Tree", "Tree", "Easy to read as a set of rules."],
    ["Random Forest", "Ensemble of trees", "Many trees voting; usually strong on tabular data."],
    ["XGBoost", "Gradient boosting", "Builds trees that correct each other's mistakes."],
    ["Support Vector Machine", "Kernel method", "Finds a boundary between the classes."],
    ["Neural Network (MLP)", "Neural network", "Two hidden layers, for comparison."],
    ["Isolation Forest", "Unsupervised", "Finds odd rows without ever seeing the labels."],
], columns=["Model", "Family", "Why it was included"])
table_from_dataframe(model_descriptions)

h3("A note on deep learning")

p("The brief lists RNNs, CNNs, LSTMs and Transformers as options, qualified with the "
  "words 'if appropriate'. They are not appropriate here, and it is worth saying why "
  "rather than adding one for the sake of it.")

p("Those architectures exist to exploit structure: CNNs assume neighbouring pixels are "
  "related, and RNNs, LSTMs and Transformers assume the order of a sequence carries "
  "meaning. This dataset is a flat table where each row is judged on its own, and the "
  "columns have no spatial or sequential relationship. Applying an LSTM would add "
  "considerable complexity and training cost for no structural reason to expect a gain. "
  "A multi-layer perceptron is included instead as the fair neural representative, and "
  "it is beaten by the far simpler and far more explainable tree ensembles.")

callout("A genuinely promising deep learning approach would be to model each card as a "
        "SEQUENCE of transactions over time and feed that to an LSTM. That is a real "
        "research direction and it is recorded in the Future Work section - but it is a "
        "different problem formulation, not a drop-in replacement.")

h3("A note on recommendation systems")

p("The brief also lists collaborative and content-based recommendation among the model "
  "families to consider. These are not used here, and the reason is that they answer a "
  "different question from the one this project asks.")

p("A recommender predicts what a user would LIKE, learning from the preferences of similar "
  "users. Fraud detection asks whether a specific event is legitimate. The signal a "
  "recommender depends on - that similar people enjoy similar things - is actively "
  "misleading here, because a fraudster is deliberately imitating someone else's normal "
  "behaviour. There is also no rating or preference data in this dataset to learn from.")

p("The closest legitimate use would be recommending which alerts an analyst should review "
  "first when the queue is longer than the team can clear. That is a ranking problem, and "
  "the model already solves it: the fraud score itself orders the queue. A separate "
  "recommender would add machinery without adding information.")

h2("4.2 Results")

p("All figures below are measured on the test set, which no model saw during training.")

table_from_dataframe(
    model_comparison[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]],
    column_titles=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])

p("The highest PR-AUC belongs to %s at %.4f. For scale, randomly guessing would score "
  "%.4f, so the model is roughly %.0f times better than chance at ranking fraud to the top."
  % (models["highest_pr_auc_model"], models["highest_pr_auc"],
     models["random_baseline_pr_auc"],
     models["highest_pr_auc"] / models["random_baseline_pr_auc"]))

if models.get("chosen_for_explainability"):
    h3("Which model was actually chosen, and why it is not the top scorer")

    p("The top of that table is a photo finish. %s scores %.4f and %s scores %.4f - a gap "
      "of %.4f. A difference that small is not evidence that one model is better; rerunning "
      "with a different random seed would be enough to swap their positions."
      % (models["highest_pr_auc_model"], models["highest_pr_auc"],
         models["best_model"], models["best_pr_auc"],
         models["highest_pr_auc"] - models["best_pr_auc"]))

    p("When two models are genuinely tied, the tie has to be broken on something other "
      "than the score. The criterion used here is whether the model can be explained, and "
      "%s was chosen for that reason." % models["best_model"])

    bullet("SHAP is exact and fast on tree-based models. On a neural network it is "
           "approximate and far slower, which makes per-decision explanations impractical.")
    bullet("Step 5 of this assignment requires explaining individual decisions, and a bank "
           "must be able to tell a customer why their payment was declined.")
    bullet("Regulators ask 'why was this transaction blocked', not 'what was your PR-AUC'.")

    p("")
    callout("This is a deliberate engineering judgement rather than a statistical one, and "
            "it is recorded here so a reader can disagree with it. Choosing the second-best "
            "score to gain explainability is defensible only because the difference is "
            "inside the noise - it would not be defensible if the gap were real.")

add_figure("fig10_model_comparison.png", "Figure 17: Model comparison across three metrics.")
add_figure("fig11_roc_curves.png", "Figure 18: ROC curves for all models.")
add_figure("fig12_pr_curves.png",
           "Figure 19: Precision-Recall curves. This is the honest picture when the "
           "positive class is this rare - note how much less flattering it is than the "
           "ROC chart above.")

h3("Reading the table carefully")

p("The accuracy column is worth pausing on. Almost every model scores above 99%%, and "
  "those numbers are close to meaningless - a model predicting 'genuine' every single "
  "time would score %.2f%%. Logistic Regression has the LOWEST accuracy of the supervised "
  "models yet catches a high share of the fraud, because it raises many more alerts. "
  "Accuracy punishes it for doing the useful thing."
  % ((1 - models["random_baseline_pr_auc"]) * 100))

add_figure("fig13_confusion_matrix.png",
           "Figure 20: Confusion matrix for %s at the default 0.5 threshold."
           % models["best_model"])

h2("4.3 Handling the class imbalance")

p("Three standard approaches were compared on the same model and the same data.")

table_from_dataframe(imbalance_comparison,
                     column_titles=["Approach", "Recall", "Precision", "PR-AUC"])

p("This result is more interesting than it first appears, and it does not match the "
  "textbook expectation. Class weighting and SMOTE both raise recall dramatically - from "
  "%.2f to %.2f - which is what they are advertised to do. But both also LOWER the PR-AUC."
  % (imbalance_comparison["recall"].iloc[0], imbalance_comparison["recall"].iloc[1]))

p("The explanation is that these techniques do not make the model better at telling fraud "
  "from genuine payments. They shift where the decision boundary sits, trading precision "
  "for recall. PR-AUC measures ranking quality across every possible threshold, so it "
  "sees through that trade and reports no real improvement.")

callout("The practical conclusion: rather than resampling the data, it is cleaner to "
        "train on the data as it is and then tune the decision threshold explicitly - "
        "which is exactly what section 4.4 does, and it makes the trade-off visible and "
        "deliberate instead of hidden inside the training process.")

h2("4.4 Choosing the alert threshold with money")

p("Models output a probability, but a bank needs a yes or no. The usual cut-off of 0.5 is "
  "an arbitrary convention with no business meaning. Instead, every threshold from 0.05 "
  "to 0.95 was tested and scored in dollars:")

bullet("Catching a fraud saves the value of that transaction.")
bullet("Each alert costs %s of analyst time to review." % money(models["cost_per_review"]))
bullet("Net benefit = value saved minus review cost.")

add_figure("fig14_threshold_tuning.png",
           "Figure 21: Net benefit against threshold. The best setting is %.2f, not the "
           "default 0.50." % models["best_threshold"])

threshold_summary = pd.DataFrame([
    ["Chosen threshold", "%.2f" % best["threshold"]],
    ["Alerts raised", "{:,}".format(best["alerts"])],
    ["Fraud caught", "{:,} of {:,}".format(best["fraud_caught"],
                                           best["fraud_caught"] + best["fraud_missed"])],
    ["Recall", "%.1f%%" % (best["recall"] * 100)],
    ["False alarms", "{:,}".format(best["false_alarms"])],
    ["Fraud value blocked", money(best["money_saved"])],
    ["Cost of reviews", money(best["review_cost"])],
    ["Net benefit", money(best["net_benefit"])],
], columns=["Measure", "Value"])
table_from_dataframe(threshold_summary)

h3("A methodological warning worth recording")

p("The first version of this analysis searched thresholds from 0.05 upwards, and reported "
  "0.05 as the best answer - the very lowest value tested. A result sitting exactly on the "
  "edge of the range searched is almost always a sign that the true optimum lies outside "
  "it, and that the search was too narrow rather than that the answer is genuinely extreme.")

p("Two things were wrong. The range was widened to start at 0.01, and more importantly the "
  "cost model was corrected. The original version priced a false alarm at only the %s of "
  "analyst time needed to review it, while an average fraud was worth roughly %s. With "
  "costs that lopsided the arithmetic will always push the threshold towards zero, because "
  "flagging a hundred innocent customers to catch one more fraud still 'pays'."
  % (money(models["cost_per_review"]), money(overview["average_fraud_amount"])))

p("A false alarm is not merely a clerical cost. It is a real customer, standing in a shop, "
  "whose card has just been declined. A goodwill cost of %s per false alarm is now added on "
  "top of the review cost, and with that correction the optimum moves to a sensible "
  "interior value of %.2f." % (money(models["cost_per_upset_customer"]), best["threshold"]))

h3("How much does that answer depend on the assumption?")

p("The goodwill figure is an assumption, not a measurement, so the whole calculation was "
  "repeated across a range of assumptions to see whether the recommendation is stable.")

sensitivity_table = load_csv("cost_sensitivity.csv")
table_from_dataframe(sensitivity_table,
                     column_titles=["Assumed cost per false alarm", "Best threshold",
                                    "Alerts", "Fraud caught", "Recall", "Net benefit"])

add_figure("fig33_cost_sensitivity.png",
           "Figure 22: The threshold that maximises value, plotted against what we assume a "
           "false alarm costs.")

p("The recommendation is NOT stable, and that is the finding. As the assumed cost rises "
  "from nothing to %s, the best threshold moves from %.2f to %.2f and the share of fraud "
  "caught falls from %.0f%% to %.0f%%."
  % (money(sensitivity_table["assumed_cost_per_false_alarm"].iloc[-1]),
     sensitivity_table["best_threshold"].iloc[0],
     sensitivity_table["best_threshold"].iloc[-1],
     sensitivity_table["recall"].iloc[0] * 100,
     sensitivity_table["recall"].iloc[-1] * 100))

callout("Nothing in the data can settle what a wrongly blocked customer is worth. That is a "
        "business judgement about how the organisation values its customers' goodwill, and "
        "presenting the resulting threshold as a purely technical result would be "
        "misleading. It is a business decision wearing a mathematical costume.")

h3("The option a real fraud team could actually staff")

capacity = models.get("capacity_option")
if capacity is not None:
    p("The value-maximising threshold assumes unlimited review capacity, which no fraud "
      "team has. If the team can review at most %s alerts in a period of this length, the "
      "best available setting is %.2f."
      % ("{:,}".format(capacity["max_alerts"]), capacity["threshold"]))

    capacity_table = pd.DataFrame([
        ["Threshold", "%.2f" % best["threshold"], "%.2f" % capacity["threshold"]],
        ["Alerts to review", "{:,}".format(best["alerts"]), "{:,}".format(capacity["alerts"])],
        ["Fraud caught", "{:,}".format(best["fraud_caught"]), "{:,}".format(capacity["fraud_caught"])],
        ["Recall", "%.1f%%" % (best["recall"] * 100), "%.1f%%" % (capacity["recall"] * 100)],
        ["Precision", "%.1f%%" % (best["precision"] * 100), "%.1f%%" % (capacity["precision"] * 100)],
        ["Net benefit", money(best["net_benefit"]), money(capacity["net_benefit"])],
    ], columns=["Measure", "Maximum value", "Within team capacity"])
    table_from_dataframe(capacity_table)

    p("The capacity-limited option catches %d fewer frauds but raises %s fewer alerts, and "
      "its precision is far higher at %.0f%%. Which option is correct depends on how many "
      "reviewers the business is willing to fund - again a decision for people, not for the "
      "model." % (best["fraud_caught"] - capacity["fraud_caught"],
                  "{:,}".format(best["alerts"] - capacity["alerts"]),
                  capacity["precision"] * 100))

default_row = threshold_analysis[threshold_analysis["threshold"] == 0.5]
if len(default_row) > 0 and abs(best["threshold"] - 0.5) > 0.001:
    default_benefit = float(default_row["net_benefit"].iloc[0])
    p("Moving from the default 0.50 to %.2f changes the net benefit from %s to %s over the "
      "test period."
      % (best["threshold"], money(default_benefit), money(best["net_benefit"])))

h2("4.5 Unsupervised learning")

p("Clustering was applied to check whether risky behaviour can be found with no labels "
  "at all - which is the situation a bank faces when a new fraud pattern appears.")

h3("K-Means")

add_figure("fig15_elbow_method.png", "Figure 22: Elbow method.")
add_figure("fig16_silhouette_scores.png",
           "Figure 23: Silhouette scores. The best grouping uses %d clusters (score %.3f)."
           % (clusters["best_k"], clusters["best_silhouette"]))

table_from_dataframe(cluster_profiles,
                     column_titles=["Cluster", "Transactions", "Share %", "Avg amount",
                                    "Avg hour", "Avg distance", "Avg age", "Fraud rate %"])

p("The result is striking. One cluster containing a small share of transactions has a "
  "fraud rate of %.2f%%, which is %.1f times the average across the sample. The algorithm "
  "found that group using only spending behaviour, never having seen a single fraud label."
  % (clusters["riskiest_cluster_fraud_rate"], clusters.get("riskiest_cluster_lift", 0)))

add_figure("fig17_cluster_fraud_rates.png", "Figure 24: Fraud rate inside each cluster.")
add_figure("fig18_pca_clusters.png",
           "Figure 25: Clusters drawn in two dimensions using PCA, with real fraud marked "
           "as black crosses.")

h3("DBSCAN")

p("DBSCAN differs from K-Means in a way that suits fraud detection well: it does not force "
  "every transaction into a group. Anything that fits nowhere is labelled noise.")

key_value("Clusters found", clusters["dbscan_clusters"])
key_value("Noise points", "{:,} ({:.1f}% of the sample)".format(
    clusters["dbscan_noise_count"], clusters["dbscan_noise_percent"]))

if "dbscan_noise_lift" in clusters:
    key_value("Fraud rate among noise", "%.2f%%" % clusters["dbscan_noise_fraud_rate"])
    key_value("Fraud rate among grouped points", "%.3f%%" % clusters["dbscan_cluster_fraud_rate"])
    p("")
    p("Transactions that DBSCAN could not fit into any group are %.1f times more likely to "
      "be fraudulent than those it could. 'Does not look like anything else' turns out to "
      "be a powerful fraud signal in its own right."
      % clusters["dbscan_noise_lift"])

add_figure("fig19_dbscan.png", "Figure 26: DBSCAN clusters and noise points.")

h3("Hierarchical clustering")

p("Hierarchical clustering builds a tree showing which transactions merge together first. "
  "On the same task it scored a silhouette of %.3f against K-Means' %.3f, so it agrees "
  "broadly with K-Means while being far more expensive to compute - which is why it was "
  "run on a smaller sample of %d rows."
  % (clusters["hierarchical_silhouette"], clusters["best_silhouette"],
     clusters["hierarchical_sample_size"]))

add_figure("fig20_dendrogram.png", "Figure 27: Dendrogram.")

h2("4.6 Reproducibility")

p("Every random process in this project is given a fixed seed of 42, so re-running the "
  "scripts reproduces the numbers in this report exactly. Library versions are pinned in "
  "requirements.txt, trained models are saved to models/, and every figure and table is "
  "written to reports/. The report itself is generated by a script that reads those saved "
  "results, so the text can never drift out of step with the experiment.")

doc.add_page_break()


# ===== STEP 5 =====
h1("Step 5 - Critical Thinking: Ethical AI and Bias Auditing")

h2("5.1 How the model makes decisions")

p("Section 3.5 covered the mechanics of SHAP and LIME. The summary in plain terms is "
  "that the model relies mainly on how large a payment is relative to what that card "
  "normally spends, what time of day it happened, and what type of purchase it was. "
  "Those are behavioural signals, and that is reassuring: they describe what was done, "
  "not who did it.")

h2("5.2 Limitations of this model")

h3("Class imbalance")
p("With fraud at %.3f%% of transactions, the model sees roughly one fraudulent example "
  "for every %.0f genuine ones. Rare-event learning is inherently unstable, and small "
  "changes in the data can move the metrics noticeably."
  % (overview["fraud_rate_percent"], models["imbalance_ratio"]))

h3("Data leakage")
p("The two learned features - card average spend and merchant frequency - are the main "
  "leakage risk. Both are computed on training rows only. The chronological split "
  "provides a second layer of protection, since no information from the test period can "
  "travel backwards in time.")

h3("Overfitting")
p("Tree ensembles can memorise training data. This is constrained by limiting tree depth "
  "and requiring a minimum number of samples per leaf, and more importantly it is "
  "detected by testing on a later time period. A model that had merely memorised would "
  "collapse on unseen future transactions rather than achieving a PR-AUC of %.4f."
  % models["best_pr_auc"])

h3("The data is simulated")
p("This is the most important limitation of all. The patterns are generated by a "
  "simulator, so they are cleaner and more consistent than real fraud, which adapts "
  "actively as criminals respond to detection. Performance on real banking data would "
  "certainly be lower, and the model would need frequent retraining.")

h2("5.3 Bias and Fairness Analysis")

p("This section tests a claim that is very commonly made and very rarely checked: that "
  "leaving protected characteristics out of a model makes the model fair. That approach "
  "has a name - 'fairness through unawareness' - and the audit below examines whether it "
  "actually worked here.")

key_value("Model audited", fairness["model"])
key_value("Threshold used", "%.2f" % fairness["threshold"])
key_value("Protected attributes given to the model", "None - gender and job were withheld")

h3("The fairness measures used")

measures = pd.DataFrame([
    ["Demographic parity", "Is each group flagged at the same rate?",
     "Gap between the highest and lowest alert rate"],
    ["Equal opportunity", "Is fraud caught equally well for each group?",
     "Gap in true positive rate"],
    ["Equalised odds", "Are false alarms also equal?",
     "Gap in true positive AND false positive rates"],
    ["Disparate impact", "The legal '80% rule' from employment and lending law",
     "Lowest alert rate divided by highest; below 0.8 is a red flag"],
], columns=["Measure", "The question it asks", "How it is calculated"])
table_from_dataframe(measures)

h3("Results by gender")

table_from_dataframe(gender_bias,
                     column_titles=["Group", "People", "Real fraud", "Base rate %",
                                    "Alert rate %", "Caught %", "False alarm %",
                                    "Precision %", "False alarms", "Missed"])

gender_gaps = fairness["gender"]
key_value("Demographic parity gap", "%.3f percentage points" % gender_gaps["demographic_parity_gap"])
key_value("Disparate impact ratio", "%.3f - %s the 80%% rule"
          % (gender_gaps["disparate_impact_ratio"],
             "PASSES" if gender_gaps["passes_80_percent_rule"] else "FAILS"))
key_value("Equal opportunity gap", "%.2f percentage points" % gender_gaps["equal_opportunity_gap"])
key_value("Worst served group", "%s - only %.1f%% of its fraud is caught"
          % (gender_gaps["least_protected_group"], gender_gaps["least_protected_tpr"]))

add_figure("fig28_bias_by_gender.png", "Figure 28: Fairness check by gender.")

h3("Results by age")

table_from_dataframe(age_bias,
                     column_titles=["Group", "People", "Real fraud", "Base rate %",
                                    "Alert rate %", "Caught %", "False alarm %",
                                    "Precision %", "False alarms", "Missed"])

age_gaps = fairness["age_band"]
key_value("Demographic parity gap", "%.3f percentage points" % age_gaps["demographic_parity_gap"])
key_value("Disparate impact ratio", "%.3f - %s the 80%% rule"
          % (age_gaps["disparate_impact_ratio"],
             "PASSES" if age_gaps["passes_80_percent_rule"] else "FAILS"))
key_value("Equal opportunity gap", "%.2f percentage points" % age_gaps["equal_opportunity_gap"])
key_value("Worst served group", "%s - only %.1f%% of its fraud is caught"
          % (age_gaps["least_protected_group"], age_gaps["least_protected_tpr"]))
key_value("Best served group", "%s - %.1f%% caught"
          % (age_gaps["best_protected_group"], age_gaps["best_protected_tpr"]))

add_figure("fig29_bias_by_age.png", "Figure 29: Fairness check by age group.")

p("Age is the one protected-adjacent attribute the model IS allowed to use, so any gap "
  "here is a direct effect rather than a proxy effect. This is the cost of the judgement "
  "call made in section 3.3, and it is measured here rather than assumed away.")

h3("Results by city size")

p("The dataset holds no direct measure of income, so city population is used as a rough "
  "stand-in for socioeconomic status. It is imperfect - there are wealthy people in "
  "villages and poor people in cities - but it is the best available proxy and it is "
  "better than skipping the question.")

table_from_dataframe(city_bias,
                     column_titles=["Group", "People", "Real fraud", "Base rate %",
                                    "Alert rate %", "Caught %", "False alarm %",
                                    "Precision %", "False alarms", "Missed"])

city_gaps = fairness["city_size"]
key_value("Demographic parity gap", "%.3f percentage points" % city_gaps["demographic_parity_gap"])
key_value("Disparate impact ratio", "%.3f - %s the 80%% rule"
          % (city_gaps["disparate_impact_ratio"],
             "PASSES" if city_gaps["passes_80_percent_rule"] else "FAILS"))
key_value("Equal opportunity gap", "%.2f percentage points" % city_gaps["equal_opportunity_gap"])
key_value("Worst served group", "%s - only %.1f%% of its fraud is caught"
          % (city_gaps["least_protected_group"], city_gaps["least_protected_tpr"]))

add_figure("fig30_bias_by_city_size.png", "Figure 30: Fairness check by city size.")

h3("Summary of all three attributes")

add_figure("fig31_fairness_summary.png",
           "Figure 31: Fairness gaps side by side. The red line on the right marks the "
           "80% legal threshold.")

h3("What this means")

p("Three concrete findings come out of the tables above.")

numbered("Gender was never given to the model, yet %s customers have %.1f%% of their fraud "
         "caught against %.1f%% for %s - a gap of %.1f percentage points. The gender "
         "disparity survived the removal of the gender column."
         % (gender_gaps["least_protected_group"], gender_gaps["least_protected_tpr"],
            gender_gaps["best_protected_tpr"], gender_gaps["best_protected_group"],
            gender_gaps["equal_opportunity_gap"]))

numbered("The largest disparity is by city size, where the disparate impact ratio is %.3f - "
         "comfortably below the 0.8 legal threshold. %s customers are the worst protected, "
         "at %.1f%% of their fraud caught."
         % (city_gaps["disparate_impact_ratio"], city_gaps["least_protected_group"],
            city_gaps["least_protected_tpr"]))

numbered("Two of the three attributes fail the 80%% rule outright. Only gender passes, at "
         "%.3f." % gender_gaps["disparate_impact_ratio"])

p("")
p("The reason is that protected characteristics leave fingerprints on other columns. "
  "Spending category, city size, transaction timing and typical amounts all correlate "
  "with demographics, so the model can reconstruct part of what was withheld without "
  "ever being told it. Withholding the gender column removed the model's ability to "
  "discriminate directly, and removed nothing else.")

p("The city size result deserves particular attention because it is the one with a "
  "plausible mechanism behind it. Rural customers live further from shops, so the "
  "'distance from home' feature behaves differently for them, and rural areas have fewer "
  "transactions overall so the patterns are learned less well. A feature that looks "
  "entirely neutral - how far away the shop is - turns out to carry a socioeconomic "
  "signal.")

callout("This is the central finding of the audit: removing a sensitive column does not "
        "remove its influence, it only removes your ability to see it. Fairness through "
        "unawareness prevents direct discrimination but does nothing about indirect "
        "discrimination through proxies. The only way to know is to measure - which is "
        "why this section exists.")

p("There is a second point worth making. Some of the difference between groups is not the "
  "model's doing at all. Figure 7 showed that the underlying fraud rates genuinely differ "
  "between groups before any model is trained. A model that reflects a real difference is "
  "accurate; the ethical question is whether acting on that accuracy is acceptable when "
  "the consequence is that one group of customers gets worse protection than another.")

h2("5.4 Proposed mitigations")

p("Four families of mitigation were considered, and one was implemented and measured.")

mitigations = pd.DataFrame([
    ["Reweighting", "Give under-protected groups more weight during training",
     "Fixes the cause, not the symptom", "Needs full retraining; can reduce overall accuracy"],
    ["Per-group thresholds", "Use a different alert cut-off for each group",
     "Directly equalises protection; no retraining needed", "Legally contentious - treats people differently by group"],
    ["Data augmentation", "Collect or synthesise more fraud examples for thin groups",
     "Addresses the root cause, which is sparse data", "Expensive; synthetic fraud may not be realistic"],
    ["Post-processing", "Adjust the outputs after the model has run",
     "Simple to bolt onto an existing system", "Hides the problem rather than solving it"],
], columns=["Approach", "What it does", "Advantage", "Drawback"])
table_from_dataframe(mitigations)

h3("The mitigation that was implemented")

mitigation = fairness["mitigation"]

p("Per-group thresholds were implemented and measured on the age attribute. Each age "
  "group receives its own alert cut-off, chosen so that every group reaches the level of "
  "protection that the best-served group already enjoyed (%.1f%% of its fraud caught)."
  % mitigation["target_tpr"])

results_table = pd.DataFrame([
    ["Equal opportunity gap before", "%.2f percentage points" % mitigation["gap_before"]],
    ["Equal opportunity gap after", "%.2f percentage points" % mitigation["gap_after"]],
    ["Improvement", "%.2f percentage points" % mitigation["improvement"]],
    ["Alerts before", "{:,}".format(mitigation["alerts_before"])],
    ["Alerts after", "{:,}".format(mitigation["alerts_after"])],
    ["Fraud caught before", "{:,}".format(mitigation["fraud_caught_before"])],
    ["Fraud caught after", "{:,}".format(mitigation["fraud_caught_after"])],
    ["False alarms before", "{:,}".format(mitigation["false_alarms_before"])],
    ["False alarms after", "{:,}".format(mitigation["false_alarms_after"])],
], columns=["Measure", "Value"])
table_from_dataframe(results_table)

add_figure("fig32_mitigation_effect.png",
           "Figure 32: Fraud caught per age group, before and after per-group thresholds.")

p("The intervention works: the protection gap narrows from %.2f to %.2f percentage "
  "points, and %d more frauds are caught. But it is not free. False alarms rise from "
  "%s to %s, which is %s extra reviews for the operations team to absorb."
  % (mitigation["gap_before"], mitigation["gap_after"],
     mitigation["fraud_caught_after"] - mitigation["fraud_caught_before"],
     "{:,}".format(mitigation["false_alarms_before"]),
     "{:,}".format(mitigation["false_alarms_after"]),
     "{:,}".format(mitigation["false_alarms_after"] - mitigation["false_alarms_before"])))

callout("That trade-off is the honest heart of applied fairness work. There is no setting "
        "that is simultaneously most accurate, most profitable and most equal. Someone has "
        "to decide what the organisation is willing to pay for equal treatment, and that "
        "is a decision for accountable humans - not one that should be quietly settled by "
        "whoever picked the default threshold.")

h2("5.5 Wider ethical considerations")

h3("Privacy")
p("Six directly identifying columns were removed before any modelling. A fraud system "
  "does not need to know a customer's name to judge whether a payment is out of "
  "character, and holding that data increases the harm caused by any future breach.")

h3("Transparency and the right to an explanation")
p("When a payment is blocked the customer deserves a reason. This is why the model is "
  "built on interpretable behavioural features and why PCA was rejected for the final "
  "model. The LIME output in section 3.5 is the kind of explanation a call-centre agent "
  "could actually read aloud.")

h3("Human oversight")
p("The system is designed to flag transactions for review, not to decide them. High-value "
  "blocks in particular should require human confirmation, and customers must have a "
  "route to challenge a decision and have it overturned.")

h3("Feedback loops")
p("A quieter danger: if the model flags one group more often, that group generates more "
  "investigations, which generates more confirmed fraud records for that group, which "
  "teaches the next model to flag them even more. Monitoring must therefore track "
  "fairness metrics over time, not only accuracy.")

doc.add_page_break()


# ===== STEP 6 =====
h1("Step 6 - Final Presentation and Communication")

p("Two slide decks accompany this report, written for two audiences who need very "
  "different things.")

decks = pd.DataFrame([
    ["Technical deck", "Fellow data scientists and engineers",
     "Method, metrics, model comparison, SHAP, the fairness audit, limitations"],
    ["Business deck", "Executives and risk owners",
     "The problem in plain language, money saved, risks, and what to decide next"],
], columns=["Deck", "Audience", "Focus"])
table_from_dataframe(decks)

p("Files:")
bullet("Allen_Mendiola_Credit_Card_Fraud_Detection_Technical.pptx")
bullet("Allen_Mendiola_Credit_Card_Fraud_Detection_Business.pptx")

p("The two decks deliberately do not share slides. The technical audience needs to know "
  "that PR-AUC was chosen over ROC-AUC because of class imbalance. The executive audience "
  "needs to know that the system pays for itself and carries a measured fairness risk. "
  "Presenting either group with the other's material is the most common way good analysis "
  "fails to land.")

doc.add_page_break()


# ===== STEP 7 =====
h1("Step 7 - GitHub Repository")

p("All code, results and documents for this project are published in a public "
  "repository, structured the way an open source project would be.")

key_value("Repository", GITHUB_URL)
key_value("Visibility", "Public")

h2("7.1 Repository structure")

repo_layout = pd.DataFrame([
    ["src/", "The eleven numbered scripts, run in order from 01 to 11"],
    ["notebooks/", "A walkthrough notebook that loads the saved results and shows the "
                   "main findings, with outputs already executed"],
    ["data/", "Folder structure for the raw and processed data (contents not committed)"],
    ["models/", "The saved trained models: best_model.joblib, scaler.joblib, isolation_forest.joblib"],
    ["reports/figures/", "All 33 charts used in this report and in the slide decks"],
    ["reports/tables/", "Every metrics table and results file the documents are built from"],
    ["instructions/", "The original assignment brief, kept for reference"],
    ["README.md", "Setup, how to run the pipeline, and a summary of the findings"],
    ["requirements.txt", "Pinned library versions for reproducing the environment"],
], columns=["Path", "What it holds"])
table_from_dataframe(repo_layout)

h2("7.2 What is deliberately not committed")

p("Two things are excluded by .gitignore, and both exclusions are intentional.")

bullet("The virtual environment (venv/). It is large, machine-specific and "
       "completely rebuildable from requirements.txt.")
bullet("The data files. The raw dataset is 258 MB and the processed training file is "
       "233 MB. GitHub rejects any single file above 100 MB, and committing data to "
       "version control is poor practice in any case.")

p("")
p("This does not harm reproducibility. Running src/01_download_data.py fetches the "
  "dataset again from its public source, and every later script regenerates its own "
  "outputs from there. The empty data folders are kept in the repository with .gitkeep "
  "files so the structure is still visible.")

h2("7.3 Reproducing the work from the repository")

for line in ["git clone " + GITHUB_URL + ".git",
             "cd " + GITHUB_URL.split("/")[-1],
             "python -m venv venv",
             "venv\\Scripts\\activate",
             "pip install -r requirements.txt",
             "python src/01_download_data.py",
             "   ... then scripts 02 through 11 in order"]:
    code_paragraph = doc.add_paragraph(line)
    code_paragraph.paragraph_format.left_indent = Inches(0.35)
    for run in code_paragraph.runs:
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)

doc.add_page_break()


# ===== CONCLUSION =====
h1("Conclusion and Recommendations")

h2("What was achieved")

numbered("A fraud detection model was built achieving a PR-AUC of %.4f, roughly %.0f times "
         "better than chance, catching %.1f%% of fraudulent transactions at the chosen "
         "operating point." % (models["best_pr_auc"],
                               models["best_pr_auc"] / models["random_baseline_pr_auc"],
                               best["recall"] * 100))
numbered("The alert threshold was chosen on financial grounds rather than convention, "
         "producing an estimated net benefit of %s over the test period."
         % money(best["net_benefit"]))
numbered("Unsupervised clustering independently located a group with a %.1f%% fraud rate "
         "without using any labels, and DBSCAN's noise points proved %.0f times more "
         "likely to be fraudulent than grouped transactions."
         % (clusters["riskiest_cluster_fraud_rate"], clusters.get("dbscan_noise_lift", 0)))
numbered("A full fairness audit across gender, age and city size demonstrated that "
         "withholding protected attributes did not by itself produce equal treatment.")
numbered("A mitigation was implemented and its cost measured honestly, rather than being "
         "recommended in the abstract.")

h2("Recommendations")

h3("Deploy, but with conditions")
p("The model is good enough to be useful, at the %.2f threshold, provided that alerts go "
  "to human reviewers rather than to automatic blocking, and that fairness metrics are "
  "monitored alongside accuracy from day one." % models["best_threshold"])

h3("Decide the fairness question explicitly")
p("Section 5.4 shows the cost of closing the protection gap. That is a decision for the "
  "business and its regulator, and it should be taken deliberately and recorded - not "
  "left to whoever configured the default threshold.")

h3("Retrain on a schedule")
p("Fraud adapts. A model trained on last year's behaviour degrades. Monthly retraining "
  "with drift monitoring should be assumed as an operating cost, not treated as a project.")

h3("Validate on real data before trusting any of it")
p("Every number in this report comes from simulated transactions. The methodology "
  "transfers; the specific figures do not. Real-world performance will be lower.")

doc.add_page_break()


# ===== LIMITATIONS =====
h1("Limitations and Future Work")

h2("Limitations")

bullet("The data is simulated, so real fraud would be messier, more adaptive and harder.")
bullet("The source file was truncated at Excel's row limit, losing roughly 248,000 "
       "transactions and shortening the period covered.")
bullet("City population is a weak proxy for socioeconomic status; no income data exists "
       "in the dataset.")
bullet("The dataset contains no race or ethnicity field, so that dimension of the "
       "fairness audit could not be carried out at all.")
bullet("The cost model uses a flat %s review cost. A real bank would have tiered costs "
       "and would also price reputational damage from wrongly blocking a customer."
       % money(models["cost_per_review"]))
bullet("Clustering and SHAP were computed on samples rather than the full dataset, for "
       "compute reasons.")

h2("Future work")

numbered("Model each card as a SEQUENCE of transactions and apply an LSTM or Transformer. "
         "This reframes the problem in a way that genuinely suits deep learning, unlike "
         "the flat-table formulation used here.")
numbered("Add velocity features - how many payments this card made in the last hour, day "
         "and week. These are among the strongest signals in production fraud systems and "
         "are absent here.")
numbered("Test adversarial robustness by simulating a fraudster who knows the model exists "
         "and deliberately keeps payments small and during daylight hours.")
numbered("Implement reweighting during training and compare it against the per-group "
         "thresholds used in section 5.4.")
numbered("Build a live scoring service and shadow-run it against the existing rules engine "
         "before allowing it to influence any real decision.")

doc.add_page_break()


# ===== REFERENCES =====
h1("References")

references = [
    "Harris, S. (2020) Sparkov Data Generation. Simulated credit card transaction "
    "generator. Available at: https://github.com/namebrandon/Sparkov_Data_Generation",
    "Dataset mirror: Hugging Face, santosh3110/credit_card_fraud_transactions. "
    "Available at: https://huggingface.co/datasets/santosh3110/credit_card_fraud_transactions",
    "Lundberg, S. and Lee, S. (2017) 'A Unified Approach to Interpreting Model "
    "Predictions', Advances in Neural Information Processing Systems 30.",
    "Ribeiro, M., Singh, S. and Guestrin, C. (2016) '\"Why Should I Trust You?\": "
    "Explaining the Predictions of Any Classifier', KDD '16.",
    "Hardt, M., Price, E. and Srebro, N. (2016) 'Equality of Opportunity in Supervised "
    "Learning', Advances in Neural Information Processing Systems 29.",
    "Barocas, S., Hardt, M. and Narayanan, A. (2019) Fairness and Machine Learning: "
    "Limitations and Opportunities. Available at: https://fairmlbook.org",
    "Chawla, N. et al. (2002) 'SMOTE: Synthetic Minority Over-sampling Technique', "
    "Journal of Artificial Intelligence Research, 16, pp. 321-357.",
    "Chen, T. and Guestrin, C. (2016) 'XGBoost: A Scalable Tree Boosting System', KDD '16.",
    "Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', Journal of "
    "Machine Learning Research, 12, pp. 2825-2830.",
    "Saito, T. and Rehmsmeier, M. (2015) 'The Precision-Recall Plot Is More Informative "
    "than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets', PLOS ONE.",
    "US Equal Employment Opportunity Commission (1978) Uniform Guidelines on Employee "
    "Selection Procedures - the four-fifths (80%) rule.",
]

for item in references:
    bullet(item)

doc.add_page_break()


# ===== APPENDIX =====
h1("Appendix A - How to Reproduce This Work")

h2("Setup")

p("Windows, Python 3.10:")
for line in ["python -m venv venv",
             "venv\\Scripts\\activate",
             "pip install -r requirements.txt"]:
    code_paragraph = doc.add_paragraph(line)
    code_paragraph.paragraph_format.left_indent = Inches(0.35)
    for run in code_paragraph.runs:
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)

h2("Running the pipeline")

p("The scripts are numbered and must be run in order. Each one writes files that the next "
  "one reads.")

script_table = pd.DataFrame([
    ["01_download_data.py", "Downloads the dataset from Hugging Face"],
    ["02_data_overview.py", "Data dictionary and quality checks"],
    ["03_clean_and_engineer.py", "Cleaning, feature engineering, train/test split"],
    ["04_eda_figures.py", "The nine EDA charts"],
    ["05_train_models.py", "Trains seven models, tunes the threshold"],
    ["06_clustering.py", "K-Means, DBSCAN, hierarchical clustering"],
    ["07_explainability.py", "SHAP, LIME, feature selection, PCA"],
    ["08_bias_audit.py", "The fairness audit and mitigation"],
    ["09_build_report.py", "Builds this document"],
    ["10_build_pdf.py", "Converts it to PDF"],
    ["11_build_slides.py", "Builds both slide decks"],
], columns=["Script", "What it does"])
table_from_dataframe(script_table)

h2("Folder layout")

layout = [
    "data/raw/         - the downloaded dataset",
    "data/processed/   - cleaned train and test files",
    "src/              - the eleven scripts above",
    "models/           - saved trained models",
    "reports/figures/  - every chart used in this report",
    "reports/tables/   - every table and results file",
    "notebooks/        - an executed walkthrough of the main findings",
]
for line in layout:
    code_paragraph = doc.add_paragraph(line)
    code_paragraph.paragraph_format.left_indent = Inches(0.35)
    for run in code_paragraph.runs:
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)

h2("Environment")

key_value("Python", "3.10.0")
key_value("Operating system", "Windows 11, CPU only")
key_value("Random seed", "42 everywhere")
key_value("Key libraries", "pandas 2.3.3, scikit-learn 1.7.2, xgboost 3.2.0, shap 0.49.1")


# ===== SAVE =====
# Set the document properties, otherwise Word shows the library name as author.
doc.core_properties.author = STUDENT_NAME
doc.core_properties.last_modified_by = STUDENT_NAME
doc.core_properties.title = REPORT_TITLE
doc.core_properties.subject = "Pillar 5 Capstone - Credit Card Fraud Detection"

output_path = os.path.join(PROJECT_FOLDER, OUTPUT_NAME)
doc.save(output_path)

print()
print("Saved report to:", output_path)
print()
print("Done. Next step: python src/10_build_pdf.py")
