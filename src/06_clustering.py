"""
Pillar 5 Capstone - Step 4: Unsupervised Learning
Groups transactions by behaviour using K-Means, DBSCAN and Hierarchical clustering.

Why bother, when we already have a supervised model?
Because in real life fraud labels arrive weeks late, or never. Clustering finds
unusual groups of behaviour without being told which rows are fraud, so it can
flag new patterns the supervised model was never trained on.

What this file produces:
    reports/figures/fig15..fig19 *.png
    reports/tables/cluster_profiles.csv
    reports/tables/clustering_results.json

Run after 05_train_models.py:
    python src/06_clustering.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage


# ===== SETTINGS =====
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(THIS_FOLDER)
PROCESSED_FOLDER = os.path.join(PROJECT_FOLDER, "data", "processed")
FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "figures")
TABLES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "tables")

RANDOM_SEED = 42

# Clustering compares every row with every other row, so it gets slow very
# quickly. We work on a random sample instead of all 838,000 rows.
CLUSTER_SAMPLE_SIZE = 20000
DENDROGRAM_SAMPLE_SIZE = 800

# The behaviours we want to group people by.
CLUSTER_FEATURES = ["amt_log", "hour", "distance_km", "age",
                    "amt_vs_card_avg", "city_pop_log"]

sns.set_theme(style="whitegrid")


# ===== STEP 1: Load and sample =====
print("=" * 60)
print("STEP 4B: UNSUPERVISED LEARNING (CLUSTERING)")
print("=" * 60)

train = pd.read_csv(os.path.join(PROCESSED_FOLDER, "train.csv"))

sample = train.sample(n=CLUSTER_SAMPLE_SIZE, random_state=RANDOM_SEED)
sample = sample.reset_index(drop=True)

print("Using a random sample of %d transactions" % len(sample))
print("Grouping by these behaviours:", ", ".join(CLUSTER_FEATURES))

# Clustering measures distances, so the columns must be on the same scale.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(sample[CLUSTER_FEATURES])

results = {}


# ===== STEP 2: K-Means - how many groups should we use? =====
# We do not know the right number of groups, so we try several and look at
# two clues: the elbow chart and the silhouette score.
print()
print("--- K-Means: finding the best number of clusters ---")

k_values = list(range(2, 11))
inertia_values = []      # how tightly packed each group is (lower is better)
silhouette_values = []   # how well separated the groups are (higher is better)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(scaled_features)

    inertia_values.append(kmeans.inertia_)

    # Silhouette is slow on lots of rows, so we measure it on part of the sample.
    score = silhouette_score(scaled_features, labels, sample_size=5000,
                             random_state=RANDOM_SEED)
    silhouette_values.append(score)

    print("  k=%2d   inertia=%10.0f   silhouette=%.3f" % (k, kmeans.inertia_, score))

# The best k by silhouette score.
best_position = int(np.argmax(silhouette_values))
best_k = k_values[best_position]
best_silhouette = silhouette_values[best_position]

print()
print("Best number of clusters by silhouette score: %d (score %.3f)" % (best_k, best_silhouette))

results["k_values"] = k_values
results["silhouette_scores"] = [round(float(v), 4) for v in silhouette_values]
results["best_k"] = int(best_k)
results["best_silhouette"] = round(float(best_silhouette), 4)

# Figure 15: the elbow chart
plt.figure(figsize=(9, 5))
plt.plot(k_values, inertia_values, marker="o", color="#4C72B0")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (lower means tighter groups)")
plt.title("Elbow method\nLook for the bend where adding groups stops helping much")
plt.xticks(k_values)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig15_elbow_method.png"), dpi=150)
plt.close()
print("  saved fig15_elbow_method.png")

# Figure 16: the silhouette chart
plt.figure(figsize=(9, 5))
bar_colours = []
for k in k_values:
    if k == best_k:
        bar_colours.append("#C44E52")
    else:
        bar_colours.append("#4C72B0")

plt.bar(k_values, silhouette_values, color=bar_colours)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score (higher means clearer groups)")
plt.title("Silhouette score for each number of clusters\n(red bar = the winner)")
plt.xticks(k_values)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig16_silhouette_scores.png"), dpi=150)
plt.close()
print("  saved fig16_silhouette_scores.png")


# ===== STEP 3: Build the final K-Means grouping and describe each group =====
print()
print("--- What does each cluster look like? ---")

final_kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
sample["cluster"] = final_kmeans.fit_predict(scaled_features)

# For each cluster, work out the average behaviour and the fraud rate.
profile_rows = []

for cluster_number in range(best_k):
    rows_in_cluster = sample[sample["cluster"] == cluster_number]

    one_profile = {}
    one_profile["cluster"] = cluster_number
    one_profile["transactions"] = len(rows_in_cluster)
    one_profile["share_percent"] = round(len(rows_in_cluster) * 100.0 / len(sample), 1)
    one_profile["avg_amount"] = round(float(rows_in_cluster["amt"].mean()), 2)
    one_profile["avg_hour"] = round(float(rows_in_cluster["hour"].mean()), 1)
    one_profile["avg_distance_km"] = round(float(rows_in_cluster["distance_km"].mean()), 1)
    one_profile["avg_age"] = round(float(rows_in_cluster["age"].mean()), 1)
    one_profile["fraud_rate_percent"] = round(float(rows_in_cluster["is_fraud"].mean() * 100), 3)

    profile_rows.append(one_profile)

profiles = pd.DataFrame(profile_rows)
profiles = profiles.sort_values("fraud_rate_percent", ascending=False)

print(profiles.to_string(index=False))
profiles.to_csv(os.path.join(TABLES_FOLDER, "cluster_profiles.csv"), index=False)

overall_fraud_rate = sample["is_fraud"].mean() * 100
riskiest_cluster = profiles.iloc[0]

print()
print("Average fraud rate across the sample : %.3f%%" % overall_fraud_rate)
print("Fraud rate in the riskiest cluster   : %.3f%%" % riskiest_cluster["fraud_rate_percent"])

if overall_fraud_rate > 0:
    lift = riskiest_cluster["fraud_rate_percent"] / overall_fraud_rate
    print("That cluster is %.1f times riskier than average," % lift)
    print("and clustering found it without ever seeing a fraud label.")
    results["riskiest_cluster_lift"] = round(float(lift), 2)

results["riskiest_cluster_fraud_rate"] = float(riskiest_cluster["fraud_rate_percent"])
results["overall_sample_fraud_rate"] = round(float(overall_fraud_rate), 4)

# Figure 17: fraud rate per cluster
plt.figure(figsize=(9, 5))
plt.bar(profiles["cluster"].astype(str), profiles["fraud_rate_percent"], color="#C44E52")
plt.axhline(y=overall_fraud_rate, color="grey", linestyle="--",
            label="Average (%.2f%%)" % overall_fraud_rate)
plt.xlabel("Cluster number")
plt.ylabel("Fraud rate (%)")
plt.title("Fraud rate inside each K-Means cluster\nSome behaviour groups are much riskier than others")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig17_cluster_fraud_rates.png"), dpi=150)
plt.close()
print("  saved fig17_cluster_fraud_rates.png")


# ===== STEP 4: See the clusters in two dimensions using PCA =====
# We have 6 behaviour columns, which we cannot draw. PCA squashes them into
# 2 columns that keep as much of the spread as possible, so we can look at them.
print()
print("--- Drawing the clusters with PCA ---")

pca = PCA(n_components=2, random_state=RANDOM_SEED)
two_dimensions = pca.fit_transform(scaled_features)

variance_kept = pca.explained_variance_ratio_.sum() * 100
print("The 2 PCA components keep %.1f%% of the original spread" % variance_kept)
results["pca_variance_kept_percent"] = round(float(variance_kept), 2)

plt.figure(figsize=(10, 7))
plt.scatter(two_dimensions[:, 0], two_dimensions[:, 1],
            c=sample["cluster"], cmap="tab10", s=4, alpha=0.5)

# Mark the real fraud cases so we can see where they sit.
fraud_positions = sample["is_fraud"] == 1
plt.scatter(two_dimensions[fraud_positions, 0], two_dimensions[fraud_positions, 1],
            c="black", s=14, marker="x", label="Actual fraud")

plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.title("Transaction clusters in 2D (K-Means, k=%d)\nBlack crosses are the real fraud cases" % best_k)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig18_pca_clusters.png"), dpi=150)
plt.close()
print("  saved fig18_pca_clusters.png")


# ===== STEP 5: DBSCAN - finds dense groups and labels the rest as noise =====
# DBSCAN is interesting for fraud because it does not force every row into a
# group. Rows that fit nowhere are marked as noise, and noise is suspicious.
print()
print("--- DBSCAN ---")

dbscan = DBSCAN(eps=0.8, min_samples=15)
dbscan_labels = dbscan.fit_predict(scaled_features)

# The label -1 means "noise", everything else is a real cluster.
number_of_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
noise_positions = dbscan_labels == -1
noise_count = int(noise_positions.sum())

print("Clusters found :", number_of_clusters)
print("Noise points   : %d (%.1f%% of the sample)" % (noise_count, noise_count * 100.0 / len(sample)))

if noise_count > 0:
    fraud_rate_in_noise = sample.loc[noise_positions, "is_fraud"].mean() * 100
    fraud_rate_in_clusters = sample.loc[~noise_positions, "is_fraud"].mean() * 100

    print("Fraud rate among noise points   : %.3f%%" % fraud_rate_in_noise)
    print("Fraud rate among grouped points : %.3f%%" % fraud_rate_in_clusters)

    results["dbscan_noise_fraud_rate"] = round(float(fraud_rate_in_noise), 4)
    results["dbscan_cluster_fraud_rate"] = round(float(fraud_rate_in_clusters), 4)

    if fraud_rate_in_clusters > 0:
        noise_lift = fraud_rate_in_noise / fraud_rate_in_clusters
        print("Transactions DBSCAN could not group are %.1f times more likely to be fraud." % noise_lift)
        results["dbscan_noise_lift"] = round(float(noise_lift), 2)

results["dbscan_clusters"] = int(number_of_clusters)
results["dbscan_noise_count"] = noise_count
results["dbscan_noise_percent"] = round(noise_count * 100.0 / len(sample), 2)

# Figure 19: DBSCAN result drawn with PCA
plt.figure(figsize=(10, 7))
plt.scatter(two_dimensions[~noise_positions, 0], two_dimensions[~noise_positions, 1],
            c=dbscan_labels[~noise_positions], cmap="tab10", s=4, alpha=0.5,
            label="Grouped")
plt.scatter(two_dimensions[noise_positions, 0], two_dimensions[noise_positions, 1],
            c="red", s=8, alpha=0.6, label="Noise (does not fit any group)")
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.title("DBSCAN: %d clusters plus %d noise points\nNoise points have a much higher fraud rate"
          % (number_of_clusters, noise_count))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig19_dbscan.png"), dpi=150)
plt.close()
print("  saved fig19_dbscan.png")


# ===== STEP 6: Hierarchical clustering =====
# This builds a tree showing which transactions merge together first.
# It is the slowest method, so we use a much smaller sample.
print()
print("--- Hierarchical clustering ---")

small_sample = sample.head(DENDROGRAM_SAMPLE_SIZE)
small_scaled = scaled_features[:DENDROGRAM_SAMPLE_SIZE]

hierarchical = AgglomerativeClustering(n_clusters=best_k)
hierarchical_labels = hierarchical.fit_predict(small_scaled)

hierarchical_silhouette = silhouette_score(small_scaled, hierarchical_labels)
print("Silhouette score with %d clusters: %.3f" % (best_k, hierarchical_silhouette))
print("(K-Means scored %.3f on the same task)" % best_silhouette)

results["hierarchical_silhouette"] = round(float(hierarchical_silhouette), 4)
results["hierarchical_sample_size"] = DENDROGRAM_SAMPLE_SIZE

# Figure 20: the dendrogram (the merge tree)
linkage_matrix = linkage(small_scaled, method="ward")

plt.figure(figsize=(11, 6))
dendrogram(linkage_matrix, truncate_mode="lastp", p=25, show_leaf_counts=True)
plt.xlabel("Groups of transactions (numbers in brackets = how many)")
plt.ylabel("Distance at which groups merge")
plt.title("Hierarchical clustering dendrogram\nCutting the tree lower down gives more groups")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "fig20_dendrogram.png"), dpi=150)
plt.close()
print("  saved fig20_dendrogram.png")


# ===== STEP 7: Save the results =====
results["sample_size"] = CLUSTER_SAMPLE_SIZE
results["features_used"] = CLUSTER_FEATURES

json_path = os.path.join(TABLES_FOLDER, "clustering_results.json")
json_file = open(json_path, "w")
json.dump(results, json_file, indent=2)
json_file.close()

print()
print("Saved results to:", json_path)
print()
print("Done. Next step: python src/07_explainability.py")
