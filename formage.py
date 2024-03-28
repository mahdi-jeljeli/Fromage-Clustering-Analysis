import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

#Chargement et description des données

# Chargement des données
data = pd.read_csv("E:\\M1_ISIDS\\Semestre_2\\Fuille de donnée\\cours\\fromage.csv", delimiter=";")

# Affichage des premières lignes du dataframe
#print(data.head())

# Description des données
#print(data.describe())

# Classification automatique (CAH et K-means)

# Exclure la première colonne (noms de fromages) lors de la sélection des caractéristiques
X = data.iloc[:, 1:].values

# Remplacer les valeurs manquantes par la moyenne des autres valeurs dans chaque colonne
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Classification avec CAH
cah = AgglomerativeClustering(n_clusters=3) # Choix arbitraire du nombre de clusters
cah_labels = cah.fit_predict(X_imputed)

# Classification avec k-Means
kmeans = KMeans(n_clusters=3) # Choix arbitraire du nombre de clusters
kmeans_labels = kmeans.fit_predict(X_imputed)

# Ajouter les étiquettes de clusters aux données
data['CAH Cluster'] = cah_labels
data['K-Means Cluster'] = kmeans_labels

# Affichage des résultats
print("Résultats de la classification avec CAH :")
print(data[['Fromages', 'CAH Cluster']])
print("\nRésultats de la classification avec k-Means :")
print(data[['Fromages', 'K-Means Cluster']])
