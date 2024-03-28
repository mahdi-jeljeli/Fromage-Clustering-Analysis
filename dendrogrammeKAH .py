import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Chargement des données
data = pd.read_csv("E:\\M1_ISIDS\\Semestre_2\\Fuille de donnée\\cours\\fromage.csv", delimiter=";")

# Exclure la première colonne (noms de fromages) lors de la sélection des caractéristiques
X = data.iloc[:, 1:].values

# Calculer la matrice de liaison
Z = linkage(X, method='ward')

# Tracer le dendrogramme
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Dendrogramme de la classification ascendante hiérarchique')
plt.xlabel('Index des échantillons')
plt.ylabel('Distance euclidienne')
plt.show()
