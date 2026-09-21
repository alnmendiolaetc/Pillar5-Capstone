"""
Pillar 5 Capstone - Step 2: Data Understanding
Looks at the raw data and produces the DATA DICTIONARY that the brief asks for.

What this file produces:
    reports/tables/data_dictionary.csv   <- the deliverable for Step 2
    reports/tables/missing_values.csv
    reports/tables/numeric_summary.csv
    reports/tables/data_overview.json    <- numbers reused later by the report builder

Run after 01_download_data.py:
    python src/02_data_overview.py
"""

import os
import json

import pandas as pd


# ===== SETTINGS =====
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(THIS_FOLDER)
RAW_FOLDER = os.path.join(PROJECT_FOLDER, "data", "raw")
TABLES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "tables")

# Show all columns when printing, otherwise pandas hides some with "..."
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)


# This is the written meaning of every column. A data dictionary is not something
# the computer can work out on its own, so we describe each variable by hand and
# then join it with the types and counts that pandas measures for us.
COLUMN_MEANINGS = {
    "trans_date_trans_time": ["Date and time the transaction happened", "text date (M/D/YY H:MM)"],
    "cc_num": ["Credit card number of the customer (identifier only)", "number"],
    "merchant": ["Name of the shop where the card was used", "text"],
    "category": ["Type of purchase, for example grocery_pos or shopping_net", "text (14 categories)"],
    "amt": ["Amount of the transaction", "US dollars"],
    "first": ["Customer first name - PERSONAL DATA", "text"],
    "last": ["Customer last name - PERSONAL DATA", "text"],
    "gender": ["Gender of the cardholder - SENSITIVE ATTRIBUTE", "M or F"],
    "street": ["Customer home street address - PERSONAL DATA", "text"],
    "city": ["Customer home city", "text"],
    "state": ["Customer home state", "text (2-letter code)"],
    "zip": ["Customer home postcode", "number"],
    "lat": ["Latitude of the customer home", "degrees"],
    "long": ["Longitude of the customer home", "degrees"],
    "city_pop": ["Population of the customer city - used as a wealth/urban proxy", "number of people"],
    "job": ["Job title of the cardholder - SENSITIVE ATTRIBUTE (social status)", "text"],
    "dob": ["Date of birth of the cardholder - used to compute age", "text date (M/D/YY, 2-digit year)"],
    "trans_num": ["Unique reference code of the transaction", "text (hash)"],
    "unix_time": ["Same transaction time but counted in seconds since 1970", "seconds"],
    "merch_lat": ["Latitude of the shop", "degrees"],
    "merch_long": ["Longitude of the shop", "degrees"],
    "is_fraud": ["TARGET: 1 means the transaction was fraud, 0 means it was genuine", "0 or 1"],
}


def find_csv_file(folder):
    """Find the biggest .csv file in a folder. The dataset is the biggest one."""
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


# ===== STEP 1: Load the data =====
print("=" * 60)
print("STEP 2: DATA UNDERSTANDING")
print("=" * 60)

if not os.path.exists(TABLES_FOLDER):
    os.makedirs(TABLES_FOLDER)

csv_path = find_csv_file(RAW_FOLDER)
print("Reading:", os.path.basename(csv_path))
print("(this takes a few seconds, the file is large)")

df = pd.read_csv(csv_path, low_memory=False)

# The file has a leftover index column from whoever saved it. Drop it.
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("Rows   :", len(df))
print("Columns:", len(df.columns))


# ===== STEP 2: What do the first rows look like? =====
print()
print("--- First 3 rows ---")
print(df.head(3))


# ===== STEP 3: Missing values =====
print()
print("--- Missing values per column ---")

missing_counts = df.isnull().sum()
missing_table = pd.DataFrame()
missing_table["column"] = missing_counts.index
missing_table["missing_count"] = missing_counts.values
missing_table["missing_percent"] = (missing_counts.values * 100.0 / len(df)).round(3)

print(missing_table.to_string(index=False))
missing_table.to_csv(os.path.join(TABLES_FOLDER, "missing_values.csv"), index=False)

total_missing = int(missing_counts.sum())
print("Total missing values in the whole table:", total_missing)


# ===== STEP 4: Duplicates =====
print()
print("--- Duplicates ---")

# A real duplicate transaction would have the same trans_num.
duplicate_rows = int(df.duplicated().sum())
duplicate_ids = int(df["trans_num"].duplicated().sum())

print("Fully identical rows      :", duplicate_rows)
print("Repeated transaction ids  :", duplicate_ids)


# ===== STEP 5: The target - how much fraud is there? =====
print()
print("--- Target variable (is_fraud) ---")

fraud_counts = df["is_fraud"].value_counts()
fraud_total = int(df["is_fraud"].sum())
fraud_rate = fraud_total * 100.0 / len(df)

print("Genuine (0):", int(fraud_counts.get(0, 0)))
print("Fraud   (1):", fraud_total)
print("Fraud rate : %.3f%%" % fraud_rate)
print()
print("NOTE: the classes are very imbalanced. This is why accuracy is a bad")
print("metric here - always predicting 'genuine' would already score %.2f%%." % (100 - fraud_rate))


# ===== STEP 6: Check the time range =====
# We check this because the row count (1,048,575) is suspiciously equal to the
# maximum number of rows Excel can open. The file may have been cut short.
print()
print("--- Time range covered ---")

# The dates are stored as text like "1/1/19 0:00", so we tell pandas the format.
dates = pd.to_datetime(df["trans_date_trans_time"], format="%m/%d/%y %H:%M")
first_date = str(dates.min())
last_date = str(dates.max())

print("Earliest transaction:", first_date)
print("Latest transaction  :", last_date)

excel_row_limit = 1048575
was_truncated = len(df) == excel_row_limit
if was_truncated:
    print()
    print("WARNING: the row count is exactly Excel's row limit.")
    print("The published file was most likely cut short when it was saved.")
    print("We will note this as a limitation in the report.")


# ===== STEP 6b: A trap in the date of birth column =====
# The birth dates use a 2-digit year, e.g. "1/19/62". Python assumes a year
# below 69 belongs to the 2000s, so 62 becomes 2062 instead of 1962. If we did
# not catch this, half the customers would appear to be born in the future and
# their ages would come out NEGATIVE.
print()
print("--- Checking the date of birth column ---")

birth_dates_naive = pd.to_datetime(df["dob"], format="%m/%d/%y")
born_in_future = int((birth_dates_naive > pd.Timestamp("2020-12-31")).sum())

print("Example raw values:", df["dob"].head(3).tolist())
print("Birth dates that a naive read puts in the FUTURE: %d (%.1f%% of rows)"
      % (born_in_future, born_in_future * 100.0 / len(df)))
print("Step 3 fixes this by subtracting 100 years from those dates.")


# ===== STEP 7: Summary of the numeric columns =====
print()
print("--- Numeric summary ---")

numeric_summary = df.describe().transpose()
print(numeric_summary)
numeric_summary.to_csv(os.path.join(TABLES_FOLDER, "numeric_summary.csv"))


# ===== STEP 8: Outliers in the transaction amount =====
# We use the IQR rule: anything far outside the middle 50% of values is unusual.
print()
print("--- Outliers in 'amt' (transaction amount) ---")

q1 = df["amt"].quantile(0.25)
q3 = df["amt"].quantile(0.75)
iqr = q3 - q1
upper_limit = q3 + 1.5 * iqr

outliers = df[df["amt"] > upper_limit]
outlier_count = len(outliers)

print("Q1 (25%%) : $%.2f" % q1)
print("Q3 (75%%) : $%.2f" % q3)
print("Upper limit for a 'normal' amount: $%.2f" % upper_limit)
print("Transactions above that limit    : %d (%.2f%%)" % (outlier_count, outlier_count * 100.0 / len(df)))
print()
print("We do NOT delete these. In fraud detection a very large payment may be")
print("exactly the thing we are trying to catch, so removing them would remove the signal.")


# ===== STEP 9: Build the DATA DICTIONARY (the Step 2 deliverable) =====
print()
print("--- Building the data dictionary ---")

dictionary_rows = []

for column_name in df.columns:
    # Look up the meaning we wrote by hand at the top of this file.
    if column_name in COLUMN_MEANINGS:
        description = COLUMN_MEANINGS[column_name][0]
        unit = COLUMN_MEANINGS[column_name][1]
    else:
        description = "(not documented)"
        unit = "(unknown)"

    unique_values = df[column_name].nunique()

    # For columns with only a few options, list them. Otherwise show the range.
    if unique_values <= 15:
        allowed = ", ".join(sorted(df[column_name].astype(str).unique()))
    elif df[column_name].dtype in ["int64", "float64"]:
        allowed = "from %s to %s" % (df[column_name].min(), df[column_name].max())
    else:
        allowed = "%d different text values" % unique_values

    one_row = {}
    one_row["variable"] = column_name
    one_row["description"] = description
    one_row["data_type"] = str(df[column_name].dtype)
    one_row["unit_or_format"] = unit
    one_row["allowed_values"] = allowed
    one_row["unique_values"] = unique_values
    one_row["missing_count"] = int(df[column_name].isnull().sum())
    one_row["example"] = str(df[column_name].iloc[0])

    dictionary_rows.append(one_row)

data_dictionary = pd.DataFrame(dictionary_rows)
dictionary_path = os.path.join(TABLES_FOLDER, "data_dictionary.csv")
data_dictionary.to_csv(dictionary_path, index=False)

print(data_dictionary[["variable", "data_type", "unit_or_format", "unique_values"]].to_string(index=False))
print()
print("Saved data dictionary to:", dictionary_path)


# ===== STEP 10: Save key numbers so the report can reuse them =====
overview = {}
overview["source_file"] = os.path.basename(csv_path)
overview["n_rows"] = int(len(df))
overview["n_columns"] = int(len(df.columns))
overview["n_fraud"] = fraud_total
overview["n_genuine"] = int(len(df) - fraud_total)
overview["fraud_rate_percent"] = round(fraud_rate, 4)
overview["total_missing_values"] = total_missing
overview["duplicate_rows"] = duplicate_rows
overview["duplicate_transaction_ids"] = duplicate_ids
overview["first_transaction"] = first_date
overview["last_transaction"] = last_date
overview["looks_truncated_by_excel"] = bool(was_truncated)
overview["dob_wrong_century_rows"] = born_in_future
overview["dob_wrong_century_percent"] = round(born_in_future * 100.0 / len(df), 2)
overview["amt_outlier_count"] = int(outlier_count)
overview["amt_outlier_upper_limit"] = round(float(upper_limit), 2)
overview["total_fraud_amount"] = round(float(df[df["is_fraud"] == 1]["amt"].sum()), 2)
overview["average_fraud_amount"] = round(float(df[df["is_fraud"] == 1]["amt"].mean()), 2)
overview["average_genuine_amount"] = round(float(df[df["is_fraud"] == 0]["amt"].mean()), 2)

json_path = os.path.join(TABLES_FOLDER, "data_overview.json")
json_file = open(json_path, "w")
json.dump(overview, json_file, indent=2)
json_file.close()

print("Saved summary numbers to:", json_path)
print()
print("Done. Next step: python src/03_clean_and_engineer.py")
