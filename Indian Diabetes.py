# -*- coding: utf-8 -*-
# Indique que le fichier utilise l'encodage UTF-8, permettant d'inclure des caractères non-ASCII

"""
Created on Thu Jun 22 15:54:24 2023
@author: bebo
# Informations de création du script et auteur
"""

from matplotlib import pyplot as plt
# Importe la bibliothèque matplotlib pour la création de graphiques
import pandas as pd
# Importe la bibliothèque pandas pour la manipulation et l'analyse de données
import numpy as np
# Importe la bibliothèque NumPy pour les opérations numériques sur des tableaux multidimensionnels

from sklearn.linear_model import LinearRegression
# Importe la classe LinearRegression de scikit-learn pour effectuer des régressions linéaires
from tensorflow.keras.models import Sequential
# Importe la classe Sequential de Keras pour créer des modèles de réseaux de neurones séquentiels
from tensorflow.keras.layers import Dense
# Importe la classe Dense de Keras pour ajouter des couches entièrement connectées à un modèle de réseau de neurones
from tensorflow.keras.utils import plot_model
# Importe la fonction plot_model de Keras pour visualiser l'architecture du modèle
from sklearn.cluster import KMeans
# Importe la classe KMeans de scikit-learn pour effectuer des clustering K-means

# Charger les données
df = pd.read_excel('indian diabetes.xlsx')
# Charge les données d'un fichier Excel dans un DataFrame pandas
print("accuracy")
print(df)
# Affiche le contenu du DataFrame

# Fonction pour tracer la droite de régression
def plot_regression_line(x, y):
    # Conversion en tableau numpy
    x = np.array(x).reshape(-1, 1)
    # Reshape x pour qu'il soit une matrice colonne, nécessaire pour scikit-learn
    y = np.array(y)
    # Convertit y en tableau numpy
    # Créer le modèle de régression linéaire
    reg = LinearRegression().fit(x, y)
    # Entraîne le modèle de régression linéaire sur les données x et y
    # Prédire les valeurs y à partir des valeurs x
    y_pred = reg.predict(x)
    # Prédit les valeurs y pour les valeurs x fournies
    # Tracer la ligne de régression
    plt.plot(x, y_pred, color='red', linewidth=2, label='Regression line')
    # Trace la ligne de régression en rouge avec une épaisseur de 2

# Clustering avec K-means
def plot_clusters(df, x_column, y_column, n_clusters):
    # Fonction pour tracer des clusters K-means

    X = df[[x_column, y_column]].dropna().values
    # Extrait les colonnes spécifiées du DataFrame et supprime les valeurs manquantes
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    # Crée et ajuste un modèle K-means avec le nombre de clusters spécifié
    df['Cluster'] = kmeans.labels_
    # Ajoute une nouvelle colonne 'Cluster' au DataFrame avec les étiquettes des clusters
    plt.figure(figsize=(20, 10))
    # Crée une nouvelle figure pour le graphique avec une taille spécifiée
    for cluster in range(n_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        # Filtre les données pour chaque cluster
        plt.scatter(cluster_data[x_column], cluster_data[y_column], label=f'Cluster {cluster}')
        # Trace les points de données pour chaque cluster avec une étiquette correspondante
    plt.xlabel(x_column)
    # Ajoute une étiquette à l'axe x
    plt.ylabel(y_column)
    # Ajoute une étiquette à l'axe y
    plt.legend()
    # Affiche la légende du graphique
    plt.show()
    # Affiche le graphique

# Afficher les nuages de points avec clustering
plot_clusters(df, 'Age', 'Glucose', 3)
# Affiche les clusters pour les colonnes 'Age' et 'Glucose' avec 3 clusters
plot_clusters(df, 'Age', 'BloodPressure', 3)
# Affiche les clusters pour les colonnes 'Age' et 'BloodPressure' avec 3 clusters
plot_clusters(df, 'Glucose', 'BloodPressure', 3)
# Affiche les clusters pour les colonnes 'Glucose' et 'BloodPressure' avec 3 clusters

# Nuage de points Age vs Glucose avec droite de régression
x = df.Age
# Extrait la colonne 'Age' du DataFrame
y = df.Glucose
# Extrait la colonne 'Glucose' du DataFrame
fig = plt.figure(figsize=(20, 10))
# Crée une nouvelle figure pour le graphique avec une taille spécifiée
plt.plot(x, y, "ob")
# Trace un nuage de points (scatter plot) avec des points bleus
plot_regression_line(x, y)
# Ajoute la ligne de régression au graphique
plt.xlabel('Age')
# Ajoute une étiquette à l'axe x
plt.ylabel('Taux Glucose')
# Ajoute une étiquette à l'axe y
plt.legend()
# Affiche la légende du graphique
plt.show()
# Affiche le graphique

# Nuage de points Age vs BloodPressure avec droite de régression
x = df.Age
# Extrait la colonne 'Age' du DataFrame
z = df.BloodPressure
# Extrait la colonne 'BloodPressure' du DataFrame
fig = plt.figure(figsize=(20, 10))
# Crée une nouvelle figure pour le graphique avec une taille spécifiée
plt.plot(x, z, "ob")
# Trace un nuage de points (scatter plot) avec des points bleus
plot_regression_line(x, z)
# Ajoute la ligne de régression au graphique
plt.xlabel('Age')
# Ajoute une étiquette à l'axe x
plt.ylabel('Pression sanguine')
# Ajoute une étiquette à l'axe y
plt.legend()
# Affiche la légende du graphique
plt.show()
# Affiche le graphique

# Nuage de points Glucose vs BloodPressure avec droite de régression
y = df.Glucose
# Extrait la colonne 'Glucose' du DataFrame
z = df.BloodPressure
# Extrait la colonne 'BloodPressure' du DataFrame
fig = plt.figure(figsize=(20, 10))
# Crée une nouvelle figure pour le graphique avec une taille spécifiée
plt.plot(y, z, "ob")
# Trace un nuage de points (scatter plot) avec des points bleus
plot_regression_line(y, z)
# Ajoute la ligne de régression au graphique
plt.xlabel('Taux Glucose')
# Ajoute une étiquette à l'axe x
plt.ylabel('Pression sanguine')
# Ajoute une étiquette à l'axe y
plt.legend()
# Affiche la légende du graphique
plt.show()
# Affiche le graphique

df.head()
# Affiche les premières lignes du DataFrame

X = df.loc[:, df.columns != 'DiabetesPresence']
# Extrait toutes les colonnes sauf 'DiabetesPresence' pour les caractéristiques (features)
Y = df.loc[:, 'DiabetesPresence']
# Extrait la colonne 'DiabetesPresence' pour les étiquettes (labels)

# Création du modèle de réseau de neurones séquentiel
model = Sequential()
# Initialise un modèle séquentiel
model.add(Dense(12, input_shape=(8,), activation='relu'))
# Ajoute une couche dense (entièrement connectée) avec 12 neurones, activation ReLU, et spécifie la forme d'entrée
model.add(Dense(8, activation='relu'))
# Ajoute une couche dense avec 8 neurones et activation ReLU
model.add(Dense(8, activation='relu'))
# Ajoute une autre couche dense avec 8 neurones et activation ReLU
model.add(Dense(8, activation='relu'))
# Ajoute encore une autre couche dense avec 8 neurones et activation ReLU
model.add(Dense(1, activation='sigmoid'))
# Ajoute une couche dense avec 1 neurone et activation sigmoïde pour la sortie binaire

# Compilation du modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Compile le modèle en spécifiant la fonction de perte, l'optimiseur, et les métriques à suivre

# Entraînement du modèle
model.fit(X, Y, epochs=100)
# Entraîne le modèle sur les données pendant 100 époques

# Évaluation du modèle
_, accuracy = model.evaluate(X, Y)
# Évalue le modèle sur l'ensemble des données
print('Accuracy: %.2f' % (accuracy*100))
# Affiche la précision du modèle en pourcentage

# Séparation des données en ensembles d'entraînement et de test
df_train = df.sample(frac=0.8)
# Sélectionne 80% des données de manière aléatoire pour l'entraînement
df_test = df.drop(df_train.index)
# Utilise le reste des données pour le test

df_train.head()
# Affiche les premières lignes de l'ensemble d'entraînement


X_train = df_train.drop(columns=['DiabetesPresence'])
Y_train = df_train['DiabetesPresence']
X_test = df_test.drop(columns=['DiabetesPresence'])
Y_test = df_test['DiabetesPresence']

# X_train = df_train.loc[:, df.columns != 'DiabetesPresence']
# # Extrait les caractéristiques de l'ensemble d'entraînement
# Y_train = df_train.loc[:, 'DiabetesPresence']
# # Extrait les étiquettes de l'ensemble d'entraînement
# X_test = df_test.loc[:, df.columns != 'DiabetesPresence']
# # Extrait les caractéristiques de l'ensemble de test
# Y_test = df_test.loc[:, 'DiabetesPresence']
# # Extrait les étiquettes de l'ensemble de test

# Visualisation du modèle
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True, show_layer_activations=True)
# Sauvegarde une visualisation de l'architecture du modèle dans un fichier image

print("Plot_model\n")
# Affiche un message de confirmation

# Re-compilation du modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Recompile le modèle avec les mêmes paramètres

# Entraînement du modèle avec validation
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=10)
# Entraîne le modèle sur l'ensemble d'entraînement avec 20% des données utilisées pour la validation, pendant 100 époques et avec une taille de batch de 10

# Visualisation de l'exactitude du modèle
plt.plot(history.history['accuracy'], color='#066b8b')
# Trace l'exactitude de l'entraînement
plt.plot(history.history['val_accuracy'], color='#b39200')
# Trace l'exactitude de la validation
plt.title('model accuracy')
# Ajoute un titre au graphique
plt.ylabel('accuracy')
# Ajoute une étiquette à l'axe des ordonnées
plt.xlabel('epoch')
# Ajoute une étiquette à l'axe des abscisses
plt.legend(['train', 'val'], loc='upper left')
# Affiche la légende
plt.show()
# Affiche le graphique

# Prédictions sur l'ensemble de test
predictions = model.predict(X_test)
# Génère les prédictions sur l'ensemble de test

predictions[0]
# Affiche la première prédiction

predictions = (model.predict(X_test) > 0.5).astype(int)
# Convertit les prédictions en valeurs binaires (0 ou 1)

# Affiche les premières prédictions avec les valeurs attendues
for i in range(5):
    print('%s => Prédit : %d,  Attendu : %d' % (X_test.iloc[i].tolist(), predictions[i], Y_test.iloc[i]))

# Évaluation du modèle sur l'ensemble de test
_, accuracy = model.evaluate(X_test, Y_test)
# Évalue le modèle sur l'ensemble de test
print('Accuracy: %.2f' % (accuracy*100))
# Affiche la précision du modèle en pourcentage

# Ajouter des couches