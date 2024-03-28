import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Chargement des données
data = pd.read_csv("E:\\M1_ISIDS\\Semestre_2\\Fuille de donnée\\cours\\fromage.csv", delimiter=";")

# Exclure la première colonne (noms de fromages) lors de la sélection des caractéristiques
X = data.iloc[:, 1:].values

# Calculer l'inertie pour différentes valeurs de k
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Tracer la courbe de l'inertie
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour le choix du nombre de clusters')
plt.show()

# Trouver le point du coude
diff_inertias = [inertias[i] - inertias[i - 1] for i in range(1, len(inertias))]
best_k = diff_inertias.index(max(diff_inertias)) + 2  # +2 car nous avons commencé à partir de 1 et diff_inertias à partir de 2
print("Le meilleur nombre de clusters détecté par la méthode du coude est :", best_k)