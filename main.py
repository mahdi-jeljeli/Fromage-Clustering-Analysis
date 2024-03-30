# Importation des modules nécessaires
import pandas as pd
from sklearn.impute import SimpleImputer

from détectionNombreAdéquatDeClasses.dendrogrammepouCAH import plot_dendrogram
from détectionNombreAdéquatDeClasses.methodedecoudepourKmeans import elbowmethod
from détectionNombreAdéquatDeClasses.silhouetteMethodeAvecCAH import silhouette_method_cah
from détectionNombreAdéquatDeClasses.silhouetteMethodeAvecK_Means import silhouette_analysis
from formage import CAH, K_Means, load_data

data = load_data("..\\project\\fromage.csv")

# Exclure la première colonne (noms de fromages) lors de la sélection des caractéristiques
X = data.iloc[:, 1:].values

# Remplacer les valeurs manquantes par la moyenne des autres valeurs dans chaque colonne
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Classification automatique (CAH et K-means)
cah_labels = CAH(X_imputed , 3)
kmeans_labels = K_Means(X_imputed , 3)

# Création du DataFrame pour l'affichage des résultats
cah_result = pd.DataFrame({'Fromages': data['Fromages'], 'CAH Cluster': cah_labels})
kmeans_result = pd.DataFrame({'Fromages': data['Fromages'], 'K-Means Cluster': kmeans_labels})

# Affichage des résultats
print("Résultats de la classification avec CAH :")
print(cah_result)

print("\nRésultats de la classification avec k-Means :")
print(kmeans_result)


# Appel de la fonction pour déterminer le meilleur nombre de clusters avec la methode de coud
best_k = elbowmethod(X_imputed)
print("Le meilleur nombre de clusters trouvé pour K_Means avec la methode de coud  est:", best_k)

# Appel de la fonction pour déterminer le meilleur nombre de clusters avec la methode de silhouette
silhouette_analysis(X_imputed , 15)

best_K_CAH_silhouette = silhouette_method_cah(X_imputed)
print("Le meilleur nombre de clusters trouvé pour la methode CAH avec la methode silhouette  est:", best_K_CAH_silhouette)

# Appel de la fonction avec le chemin du fichier CSV
plot_dendrogram(X_imputed)
