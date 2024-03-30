
from sklearn.cluster import AgglomerativeClustering, KMeans
import pandas as pd
# Chargement des données
def load_data(data_path):
    data = pd.read_csv(data_path, delimiter=";")
    return data
# Classification avec CAH
def CAH(data, n_clusters):
    cah = AgglomerativeClustering(n_clusters=n_clusters)
    cah_labels = cah.fit_predict(data)
    return cah_labels

# Classification avec k-Means
def K_Means(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_labels = kmeans.fit_predict(data)
    return kmeans_labels
