from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# performs kmeans clustering for the average rating and the number of ratings per authors and saves the plot
# code from Week 6 - Clustering and PCA Workshop
def plot_kmeans(df, clusters):
    colormap = {0: 'tomato', 1: 'mediumseagreen', 2: 'lightskyblue'}
    handles = [plt.Line2D([], [], marker='o', color=color, linestyle='None') for color in colormap.values()]
    labels = [f'Cluster {cluster_id}' for cluster_id in colormap.keys()]

    plt.figure(figsize=(7, 10))
    plt.scatter(df['Num-Ratings'], df['Average-Rating'],
               c=[colormap.get(x) for x in clusters.labels_])

    plt.xlabel('Total Ratings')
    plt.ylabel('Average Rating')
    plt.grid(axis='y')
    plt.title("K-MeansClustering of Author Rating Quantity and Average Rating")
    plt.legend(handles, labels, loc='upper right')
    plt.savefig('../../plots/kmeans-no-outliers.png')
    plt.show()

# performs elbow method to find optimal k clusters and saves the plot
# code from Week 6 - Clustering and PCA Workshop
def elbow_method(df):
    normalized_data = MinMaxScaler().fit_transform(df)

    distortions = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(normalized_data)
        distortions.append(kmeans.inertia_)

    plt.plot(k_range, distortions, 'bx-')

    plt.title('The Elbow Method showing the optimal k')
    plt.xlabel('k')
    plt.ylabel('Distortion')

    plt.savefig('../../plots/elbow.png')
    plt.show()
