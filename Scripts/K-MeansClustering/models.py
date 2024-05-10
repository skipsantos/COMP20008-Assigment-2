import pandas as pd
import model_functions as mf
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

ratings_df = pd.read_csv('../Data-Files/raw-files/BX-Ratings.csv')
books_df = pd.read_csv('../Data-Files/preprocessed-files/Preprocessed_Books.csv')

merged_df = ratings_df.merge(books_df, on='ISBN')

# Generating columns for average ratings and number of ratings per author
author_ratings = merged_df.groupby('Author-Tokens').agg({'Book-Rating': ['mean', 'count']})
author_ratings.columns = ['Average-Rating', 'Num-Ratings']
q = author_ratings["Num-Ratings"].quantile(0.995)
author_ratings = author_ratings[author_ratings['Num-Ratings'] < q]


# Use elbow method to find optimal clusters
mf.elbow_method(author_ratings)
# k found = 3

# Kmeans cluster initialisation and plotting
clusters = KMeans(n_clusters=3)
clusters.fit(author_ratings)
labels = clusters.labels_

mf.plot_kmeans(author_ratings, clusters)
author_ratings["Cluster"] = labels

# Generating descriptive statistics for each cluster
clusters_info = {}
for cluster_id in range(clusters.n_clusters):
    cluster_points = author_ratings[labels == cluster_id]
    cluster_mean = np.mean(cluster_points, axis=0)
    cluster_median = np.median(cluster_points, axis=0)
    cluster_max = np.max(cluster_points, axis=0)
    cluster_min = np.min(cluster_points, axis=0)
    num_points = len(cluster_points)
    cluster_variance = np.var(cluster_points, axis=0)

    clusters_info[cluster_id] = {
        'Mean': cluster_mean,
        'Median': cluster_median,
        'Number of Points': num_points,
        'Variance': cluster_variance,
        'Max': cluster_max,
        'Min': cluster_min
    }

for cluster_id, info in clusters_info.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Mean: {info['Mean']}")
    print(f"  Median: {info['Median']}")
    print(f"  Number of Points: {info['Number of Points']}")
    print(f"  Variance: {info['Variance']}")
    print(f"  Max: {info['Max']}")
    print(f"  Min: {info['Min']}")

author_ratings.to_csv('../Data-Files/Author-Ratings-NoOutliers.csv')

author_ratings.reset_index(inplace=True)


