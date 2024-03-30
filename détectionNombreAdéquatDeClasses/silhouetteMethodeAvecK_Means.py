from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
def silhouette_analysis(data, max_clusters):
    """
    Perform silhouette analysis to determine the optimal number of clusters using K-Means.

    Parameters:
    - data (numpy.ndarray): Array of input data.
    - max_clusters (int): Maximum number of clusters to consider.

    Returns:
    - best_k (int): Optimal number of clusters.
    """

    best_score = -1
    best_k = -1

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        #print(f"For n_clusters = {k}, the average silhouette_score is : {silhouette_avg}")

        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_k = k

    print(f"The best number of clusters of K_Means is: {best_k} with a silhouette score of {best_score}")
    return best_k
