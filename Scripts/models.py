import pandas as pd
import model_functions as mf
from sklearn.cluster import KMeans

ratings_df = pd.read_csv('data-files/raw-files/BX-Ratings.csv')
books_df = pd.read_csv('data-files/preprocessed-files/Preprocessed_Books.csv')

merged_df = ratings_df.merge(books_df, on='ISBN')

# Generating columns for average ratings and number of ratings per author
author_ratings = merged_df.groupby('Author-Tokens').agg({'Book-Rating': ['mean', 'count']})
author_ratings.columns = ['Average-Rating', 'Num-Ratings']
author_ratings.to_csv('data-files/Author-Ratings.csv')

# Use elbow method to find optimal clusters
mf.elbow_method(author_ratings)
# k found = 3

# Kmeans cluster initialisation and plotting
clusters = KMeans(n_clusters=3)
clusters.fit(author_ratings)
author_ratings['Cluster'] = clusters
mf.plot_kmeans(author_ratings, clusters)

author_ratings.reset_index(inplace=True)


