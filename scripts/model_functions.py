from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# performs kmeans clustering for the average rating and the number of ratings per authors and saves the plot
# code from Week 6 - Clustering and PCA Workshop
def plot_kmeans(df, clusters):
    colormap = {0: 'tomato', 1: 'mediumseagreen', 2: 'lightskyblue'}

    plt.figure(figsize=(7, 10))
    plt.scatter(df['Average-Rating'], df['Num-Ratings'],
               c=[colormap.get(x) for x in clusters.labels_])

    plt.xlabel('Average Rating')
    plt.ylabel('Num Ratings')
    plt.title(f"k = {len(set(clusters.labels_))}")

    plt.show()
    plt.savefig('../Plots/kmeans.png')


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

    plt.savefig('../plots/elbow.png')
    plt.show()
