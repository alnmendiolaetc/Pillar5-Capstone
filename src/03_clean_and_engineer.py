"""
Pillar 5 Capstone - Step 3: Data Preprocessing and Feature Engineering
Cleans the raw data, builds new useful columns, and splits it into train and test.

What this file produces:
    data/processed/train.csv
    data/processed/test.csv
    reports/tables/feature_list.json
    reports/tables/feature_engineering_notes.csv

Run after 02_data_overview.py:
    python src/03_clean_and_engineer.py
"""

import os
import json

import numpy as np
import pandas as pd


# ===== SETTINGS =====
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(THIS_FOLDER)
RAW_FOLDER = os.path.join(PROJECT_FOLDER, "data", "raw")
PROCESSED_FOLDER = os.path.join(PROJECT_FOLDER, "data", "processed")
TABLES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "tables")

# We keep the newest 20% of transactions as the test set.
TEST_PORTION = 0.20

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)


def find_csv_file(folder):
    """Find the biggest .csv file in a folder."""
    biggest_name = ""
    biggest_size = 0
    for file_name in os.listdir(folder):
        if file_name.lower().endswith(".csv"):
            full_path = os.path.join(folder, file_name)
            size = os.path.getsize(full_path)
            if size > biggest_size:
                biggest_size = size
                biggest_name = full_path
    return biggest_name


def distance_in_km(lat1, lon1, lat2, lon2):
    """
    Work out the distance between two points on the Earth.
    This is the haversine formula. We use it to measure how far the shop was
    from the customer's home, because a payment far from home is more suspicious.
    """
    earth_radius_km = 6371.0

    # The formula needs radians, not degrees.
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lat_difference = np.radians(lat2 - lat1)
    lon_difference = np.radians(lon2 - lon1)

    a = np.sin(lat_difference / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(lon_difference / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_km * c


# ===== STEP 1: Load the raw data =====
print("=" * 60)
print("STEP 3: CLEANING AND FEATURE ENGINEERING")
print("=" * 60)

for folder in [PROCESSED_FOLDER, TABLES_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

csv_path = find_csv_file(RAW_FOLDER)
print("Reading:", os.path.basename(csv_path))

df = pd.read_csv(csv_path, low_memory=False)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("Loaded", len(df), "rows")


# ===== STEP 2: Clean the data =====
print()
print("--- Cleaning ---")

# 2a. Remove duplicate transactions (same reference code appearing twice).
rows_before = len(df)
df = df.drop_duplicates(subset=["trans_num"])
print("Removed %d duplicate transactions" % (rows_before - len(df)))

# 2b. Turn the text dates into real date values so we can do maths with them.
# Both columns are written as month/day/year with only TWO digits for the year,
# for example "1/19/62". We tell pandas the exact format instead of letting it
# guess, because guessing is slow and can be wrong.
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"],
                                             format="%m/%d/%y %H:%M")
df["dob"] = pd.to_datetime(df["dob"], format="%m/%d/%y")

# 2c. Fix the century in the date of birth.
# A two-digit year is ambiguous: does "62" mean 1962 or 2062? Python assumes
# anything below 69 is in the 2000s, so "1/19/62" became 2062 - a birthday in
# the future, which then produced NEGATIVE ages. Nobody in this data was born
# after the transactions happened, so any birth date that lands in the future
# must really be 100 years earlier.
born_in_the_future = df["dob"] > pd.Timestamp("2020-12-31")
print("Fixed %d birth dates that were read as the wrong century" % born_in_the_future.sum())
df.loc[born_in_the_future, "dob"] = df.loc[born_in_the_future, "dob"] - pd.DateOffset(years=100)

# 2d. Drop rows where the target is missing - we cannot learn from those.
rows_before = len(df)
df = df.dropna(subset=["is_fraud"])
print("Removed %d rows with no target value" % (rows_before - len(df)))

# 2e. Sort by time. This matters because we split by time later on.
df = df.sort_values("trans_date_trans_time")
df = df.reset_index(drop=True)

print("Rows after cleaning:", len(df))


# ===== STEP 3: Build new features =====
print()
print("--- Creating new features ---")

# 3a. Age of the cardholder at the time of the transaction.
age_in_days = (df["trans_date_trans_time"] - df["dob"]).dt.days
df["age"] = age_in_days / 365.25
print("  age               - how old the cardholder was")

# 3b. Time features. Fraud often happens at unusual hours.
df["hour"] = df["trans_date_trans_time"].dt.hour
df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
df["month"] = df["trans_date_trans_time"].dt.month
print("  hour, day_of_week, month - when the payment happened")

# 3c. Night flag. 10pm to 4am is when the card owner is usually asleep.
df["is_night"] = 0
df.loc[(df["hour"] >= 22) | (df["hour"] <= 4), "is_night"] = 1
print("  is_night          - 1 if the payment was between 10pm and 4am")

# 3d. How far the shop was from the customer's home.
df["distance_km"] = distance_in_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
print("  distance_km       - distance from home to the shop")

# 3e. The amount is very skewed (most payments small, a few huge).
# Taking the logarithm squashes it so models handle it better.
df["amt_log"] = np.log1p(df["amt"])
print("  amt_log           - amount on a log scale")

# 3f. City population is also very skewed, so we log it too.
df["city_pop_log"] = np.log1p(df["city_pop"])
print("  city_pop_log      - city size on a log scale")


# ===== STEP 4: Split into train and test BY TIME =====
# We train on older transactions and test on newer ones. This copies real life:
# a bank builds a model on the past and uses it on payments that have not happened yet.
# A random split would let the model peek at the future, which flatters the score.
print()
print("--- Splitting into train and test by date ---")

split_position = int(len(df) * (1 - TEST_PORTION))
split_date = df["trans_date_trans_time"].iloc[split_position]

train = df[df["trans_date_trans_time"] < split_date].copy()
test = df[df["trans_date_trans_time"] >= split_date].copy()

print("Split date :", split_date)
print("Train rows : %d  (fraud: %d, %.3f%%)" % (len(train), train["is_fraud"].sum(),
                                                train["is_fraud"].mean() * 100))
print("Test rows  : %d  (fraud: %d, %.3f%%)" % (len(test), test["is_fraud"].sum(),
                                                test["is_fraud"].mean() * 100))


# ===== STEP 5: Features that learn from the data =====
# IMPORTANT: these are worked out on the TRAINING data only, then applied to both.
# If we used the test data to build them, information from the future would leak
# into the model and our scores would be a lie.
print()
print("--- Features learned from the training set only ---")

# 5a. Average spend of each card, and how far this payment is from that average.
card_average = train.groupby("cc_num")["amt"].mean()
overall_average = train["amt"].mean()

train["card_avg_amt"] = train["cc_num"].map(card_average)
test["card_avg_amt"] = test["cc_num"].map(card_average)

# A card in the test set that we never saw in training gets the overall average.
train["card_avg_amt"] = train["card_avg_amt"].fillna(overall_average)
test["card_avg_amt"] = test["card_avg_amt"].fillna(overall_average)

train["amt_vs_card_avg"] = train["amt"] / train["card_avg_amt"]
test["amt_vs_card_avg"] = test["amt"] / test["card_avg_amt"]
print("  amt_vs_card_avg   - how many times bigger than this card's normal spend")

# 5b. How common each shop is. Rare shops can be riskier.
merchant_counts = train["merchant"].value_counts()
train["merchant_freq"] = train["merchant"].map(merchant_counts).fillna(0)
test["merchant_freq"] = test["merchant"].map(merchant_counts).fillna(0)
print("  merchant_freq     - how often this shop appears in training data")


# ===== STEP 6: Turn the purchase category into numbers =====
# Models cannot read text, so we make one yes/no column per category.
# This is called one-hot encoding.
print()
print("--- Encoding the 'category' column ---")

train_categories = pd.get_dummies(train["category"], prefix="cat")
test_categories = pd.get_dummies(test["category"], prefix="cat")

# Make sure test has exactly the same columns as train, in the same order.
test_categories = test_categories.reindex(columns=train_categories.columns, fill_value=0)

train = pd.concat([train, train_categories], axis=1)
test = pd.concat([test, test_categories], axis=1)

print("Created %d category columns" % len(train_categories.columns))


# ===== STEP 7: Build grouping columns for the bias audit (Step 5) =====
# These are NOT given to the model. They are only used later to check whether
# the model treats different groups of people differently.
print()
print("--- Creating group labels for the fairness check ---")

for table in [train, test]:
    table["age_band"] = pd.cut(table["age"],
                               bins=[0, 30, 45, 60, 200],
                               labels=["Under 30", "30-44", "45-59", "60 and over"])

    table["city_size"] = pd.cut(table["city_pop"],
                                bins=[0, 2500, 50000, 500000, 50000000],
                                labels=["Rural", "Small town", "City", "Large city"])

print("  age_band   - Under 30 / 30-44 / 45-59 / 60 and over")
print("  city_size  - Rural / Small town / City / Large city")
print("  gender     - kept from the original data")


# ===== STEP 8: Choose the final list of model features =====
# Note on ethics: gender and job are deliberately LEFT OUT of the model.
# Banks are not allowed to base decisions on protected characteristics.
# We still keep those columns in the file so Step 5 can check the model for bias.
# (Step 5 will show that leaving them out does not automatically make it fair.)
FEATURE_COLUMNS = [
    "amt",
    "amt_log",
    "amt_vs_card_avg",
    "card_avg_amt",
    "merchant_freq",
    "age",
    "hour",
    "day_of_week",
    "month",
    "is_night",
    "distance_km",
    "city_pop",
    "city_pop_log",
]

# Add the one-hot category columns to the feature list.
for column_name in train_categories.columns:
    FEATURE_COLUMNS.append(column_name)

# Columns we keep for the fairness audit and for reporting, but never train on.
EXTRA_COLUMNS = ["is_fraud", "gender", "age_band", "city_size", "job",
                 "category", "trans_date_trans_time"]

print()
print("--- Columns we deliberately removed ---")
print("  first, last, street, trans_num, cc_num, zip")
print("  Reason: these are personal details that identify a real person.")
print("  Keeping them would be a privacy risk and they do not help predict fraud.")
print()
print("  gender, job")
print("  Reason: protected characteristics must not drive the decision.")

print()
print("Total model features:", len(FEATURE_COLUMNS))


# ===== STEP 9: Save the processed data =====
print()
print("--- Saving ---")

columns_to_save = FEATURE_COLUMNS + EXTRA_COLUMNS

train_out = train[columns_to_save]
test_out = test[columns_to_save]

train_path = os.path.join(PROCESSED_FOLDER, "train.csv")
test_path = os.path.join(PROCESSED_FOLDER, "test.csv")

train_out.to_csv(train_path, index=False)
test_out.to_csv(test_path, index=False)

print("Saved:", train_path, "(%d rows)" % len(train_out))
print("Saved:", test_path, "(%d rows)" % len(test_out))

# Save the feature list so the later scripts use exactly the same columns.
feature_info = {}
feature_info["feature_columns"] = FEATURE_COLUMNS
feature_info["extra_columns"] = EXTRA_COLUMNS
feature_info["split_date"] = str(split_date)
feature_info["n_train"] = int(len(train_out))
feature_info["n_test"] = int(len(test_out))
feature_info["train_fraud_rate_percent"] = round(float(train["is_fraud"].mean() * 100), 4)
feature_info["test_fraud_rate_percent"] = round(float(test["is_fraud"].mean() * 100), 4)
feature_info["dropped_pii"] = ["first", "last", "street", "trans_num", "cc_num", "zip"]
feature_info["dropped_protected"] = ["gender", "job"]

json_path = os.path.join(TABLES_FOLDER, "feature_list.json")
json_file = open(json_path, "w")
json.dump(feature_info, json_file, indent=2)
json_file.close()

print("Saved:", json_path)


# ===== STEP 10: Write down why each feature exists =====
# The brief asks for justifications, so we save them as a table.
notes = [
    ["amt", "Original", "The payment amount. Fraudsters often test with small amounts then spend big."],
    ["amt_log", "Engineered", "Log of the amount. Squashes a very skewed range so models cope better."],
    ["amt_vs_card_avg", "Engineered", "Amount divided by this card's usual spend. Catches out-of-character payments."],
    ["card_avg_amt", "Engineered", "The card's normal spending level, learned from training data only."],
    ["merchant_freq", "Engineered", "How often the shop appears. Unknown shops carry more risk."],
    ["age", "Engineered", "Age at the time of payment, worked out from date of birth."],
    ["hour", "Engineered", "Hour of the day. Fraud peaks at night."],
    ["day_of_week", "Engineered", "Day of the week, to capture weekly habits."],
    ["month", "Engineered", "Month, to capture seasonal effects."],
    ["is_night", "Engineered", "1 if between 10pm and 4am, when the real owner is usually asleep."],
    ["distance_km", "Engineered", "Distance from home to the shop, using the haversine formula."],
    ["city_pop", "Original", "Size of the customer's city, a rough proxy for urban or rural."],
    ["city_pop_log", "Engineered", "Log of city size, because the raw values are very skewed."],
    ["cat_*", "Engineered", "One yes/no column per purchase category (one-hot encoding)."],
]

notes_table = pd.DataFrame(notes, columns=["feature", "type", "why_we_use_it"])
notes_path = os.path.join(TABLES_FOLDER, "feature_engineering_notes.csv")
notes_table.to_csv(notes_path, index=False)

print("Saved:", notes_path)
print()
print("Done. Next step: python src/04_eda_figures.py")
