# Detecting Fraudulent Credit Card Transactions

**Pillar 5 Capstone — End-to-End Machine Learning Lifecycle**
Allen Mendiola · Finance domain · Fraud detection

An end-to-end machine learning project: framing the business problem, understanding and
cleaning the data, engineering features, training and comparing seven models, explaining
what the winning model learned, and auditing it for bias against customer groups.

Repository: <https://github.com/alnmendiolaetc/Pillar5-Capstone>

---

## Headline results

| | |
|---|---|
| Dataset | 1,048,575 card transactions, 6,006 of them fraudulent (0.573%) |
| Best model | Random Forest — PR-AUC **0.8605** |
| Random baseline | PR-AUC 0.0055 (so roughly **156× better than chance**) |
| At the chosen threshold | 88% of fraud caught, 56% of alerts genuine |
| Estimated net benefit | ~$555,000 over the test period |

Accuracy is deliberately **not** the headline. With fraud at 0.573% of transactions, a model
that predicts "genuine" every single time scores 99.4% accuracy and catches nothing.

---

## How to run it

Windows, Python 3.10:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then run the scripts **in order** — each writes files the next one reads:

```bash
python src/01_download_data.py        # downloads the dataset (~110 MB)
python src/02_data_overview.py        # data dictionary + quality checks
python src/03_clean_and_engineer.py   # cleaning, features, train/test split
python src/04_eda_figures.py          # the nine EDA charts
python src/05_train_models.py         # trains seven models, tunes the threshold
python src/06_clustering.py           # K-Means, DBSCAN, hierarchical
python src/07_explainability.py       # SHAP, LIME, feature selection, PCA
python src/08_bias_audit.py           # fairness audit + mitigation
python src/09_build_report.py         # builds the Word report
python src/10_build_pdf.py            # converts it to PDF
python src/11_build_slides.py         # builds both slide decks
```

Total runtime is roughly 20–30 minutes on a 4-core CPU.

---

## Layout

```
├── data/
│   ├── raw/            downloaded dataset
│   └── processed/      cleaned train/test files
├── src/                the eleven scripts above
├── models/             saved trained models (.joblib)
├── reports/
│   ├── figures/        every chart (33 PNGs)
│   └── tables/         every metrics table and results file
├── instructions/       the original assignment brief
└── requirements.txt
```

---

## The data

**Sparkov simulated credit card transactions** — 1,048,575 rows × 22 columns, covering
2019-01-01 to 2020-03-10. Downloaded from Hugging Face
(`santosh3110/credit_card_fraud_transactions`), no account needed.

This dataset was chosen over the better-known ULB "creditcard" set for one specific reason:
it keeps `gender`, `dob`, `job` and `city_pop`. The ULB set is anonymised into unnamed
components V1–V28, which would have made both the SHAP explanations and the mandatory bias
audit impossible to carry out meaningfully.

### Two data quality problems found

**1. The file is truncated.** It contains exactly 1,048,575 rows — precisely Excel's row
limit. The original Sparkov file has 1,296,675 rows, so roughly 248,000 transactions were
lost before publication.

**2. Birth dates were being read in the wrong century.** Dates are stored as `M/D/YY`, and
Python treats a two-digit year below 69 as belonging to the 2000s — so `1/19/62` was read
as **2062**, not 1962. This affected over half the rows and produced customers born in the
future with **negative ages**.

It was caught only because a clustering profile reported an average customer age of **minus
41 years**. Had it gone unnoticed, the age fairness audit would have been auditing nonsense
while appearing to work perfectly. `src/03` now fixes the century explicitly.

---

## Approach notes

**Chronological split, not random.** Training uses the earlier transactions and testing the
later ones. A random split would scatter transactions from the same week across both sides,
letting the model learn from the future to predict the past.

**Leakage control.** The two learned features — each card's average spend and each merchant's
frequency — are computed on training rows only, then applied to test rows.

**Privacy.** Name, street, card number, postcode and transaction reference are dropped before
modelling. A fraud model does not need to know who someone is to judge whether a payment is
out of character.

**Protected attributes withheld.** Gender and job are never given to the model. They are kept
aside purely so the fairness audit can measure the model's behaviour across those groups.

---

## Three findings worth reading

**Class weights and SMOTE both *lowered* PR-AUC.** Recall jumped from 0.26 to 0.90, exactly as
advertised — but ranking quality did not improve. These techniques move the decision boundary;
they do not make the model wiser. Tuning the threshold explicitly is cleaner, because it makes
the trade-off visible instead of hiding it inside training.

**The "best" threshold is a business judgement in disguise.** As the assumed cost of wrongly
blocking a genuine customer rises from $0 to $250, the optimal threshold moves from 0.25 to
0.75. Nothing in the data settles what that cost is.

**Withholding gender did not make the model gender-blind.** Protected characteristics leave
fingerprints on spending category, city size and transaction timing, so the model reconstructs
part of what was withheld. Removing a sensitive column removes your ability to *see* its
effect, not the effect itself. Per-group thresholds close most of the gap — at the price of
more false alarms.

---

## Deliverables

- `Allen_Mendiola_Credit_Card_Fraud_Detection.docx` / `.pdf` — the full written report
- `Allen_Mendiola_Credit_Card_Fraud_Detection_Technical.pptx` — 15 slides for a technical audience
- `Allen_Mendiola_Credit_Card_Fraud_Detection_Business.pptx` — 8 slides for executives

---

## Reproducibility

Every random process uses seed **42**. Library versions are pinned in `requirements.txt`.
The report and slides are *generated* by scripts that read the saved results files, so the
numbers in the documents can never drift out of step with the experiment that produced them.

Environment: Python 3.10.0, Windows 11, CPU only.
Key libraries: pandas 2.3.3, scikit-learn 1.7.2, xgboost 3.2.0, shap 0.49.1.
