import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

def plot_dendrogram(data):

    # Calculer la matrice de liaison
    Z = linkage(data, method='ward')

    # Tracer le dendrogramme
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title('Dendrogramme de la classification ascendante hiérarchique')
    plt.xlabel('Index des échantillons')
    plt.ylabel('Distance euclidienne')
    plt.show()
