
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def elbowmethod(data):
    # Calculer l'inertie pour différentes valeurs de k
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
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
    return best_k

