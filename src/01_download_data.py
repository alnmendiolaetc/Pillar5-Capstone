"""
Pillar 5 Capstone - Step 2: Data Collection
Downloads the credit card fraud dataset and saves it into the data/raw folder.

Dataset: "Credit Card Transactions Fraud Detection" (Sparkov simulator)
Source : https://huggingface.co/datasets/santosh3110/credit_card_fraud_transactions
Licence: public dataset, free to download, no login needed.

Run this file first:
    python src/01_download_data.py
"""

import os
import zipfile

import requests


# ===== SETTINGS =====
# The web address of the dataset file.
DATA_URL = "https://huggingface.co/datasets/santosh3110/credit_card_fraud_transactions/resolve/main/credit_card_fraud_transactions.zip"

# Where we want to keep the data.
# We build the paths from this file's location so the script works
# no matter which folder you run it from.
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(THIS_FOLDER)
RAW_FOLDER = os.path.join(PROJECT_FOLDER, "data", "raw")
ZIP_PATH = os.path.join(RAW_FOLDER, "credit_card_fraud_transactions.zip")


def download_file(url, save_path):
    """Download a file from the internet and show the progress."""

    # stream=True lets us download the file in small pieces instead of
    # loading the whole 110 MB into memory at once.
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    # The server tells us the total size so we can show a percentage.
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    output_file = open(save_path, "wb")
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        output_file.write(chunk)
        downloaded = downloaded + len(chunk)

        if total_size > 0:
            percent = downloaded * 100 / total_size
            print("   downloaded %.1f%% (%.1f MB)" % (percent, downloaded / 1024 / 1024))

    output_file.close()


# ===== STEP 1: Make sure the folder exists =====
print("=" * 60)
print("STEP 2: DATA COLLECTION")
print("=" * 60)

if not os.path.exists(RAW_FOLDER):
    os.makedirs(RAW_FOLDER)
    print("Created folder:", RAW_FOLDER)


# ===== STEP 2: Download the zip file =====
# If the file is already downloaded we skip it, so re-running is fast.
if os.path.exists(ZIP_PATH):
    print("Zip file already exists, skipping download.")
else:
    print("Downloading the dataset...")
    print("URL:", DATA_URL)
    download_file(DATA_URL, ZIP_PATH)
    print("Download finished.")


# ===== STEP 3: Unzip it =====
print()
print("Opening the zip file...")

zip_file = zipfile.ZipFile(ZIP_PATH, "r")
names_inside = zip_file.namelist()

print("Files inside the zip:")
for name in names_inside:
    print("   -", name)

zip_file.extractall(RAW_FOLDER)
zip_file.close()

print("Extracted into:", RAW_FOLDER)


# ===== STEP 4: Show what we ended up with =====
print()
print("Files now in data/raw:")

all_files = os.listdir(RAW_FOLDER)
for file_name in all_files:
    full_path = os.path.join(RAW_FOLDER, file_name)
    size_mb = os.path.getsize(full_path) / 1024 / 1024
    print("   %-45s %8.1f MB" % (file_name, size_mb))

print()
print("Done. Next step: python src/02_data_overview.py")
