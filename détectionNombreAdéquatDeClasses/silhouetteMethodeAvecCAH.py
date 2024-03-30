from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering


def silhouette_method_cah(data):
    silhouette_scores = []
    for k in range(2, 11):  # Tester différents nombres de clusters de 2 à 10
        cah = AgglomerativeClustering(n_clusters=k)
        cluster_labels = cah.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    best_k = silhouette_scores.index(max(silhouette_scores)) + 2  # Ajouter 2 car nous avons commencé à partir de k=2
    return best_k
