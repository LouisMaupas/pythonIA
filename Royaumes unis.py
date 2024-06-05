# -*- coding: utf-8 -*-
# Indique que le fichier utilise l'encodage UTF-8, permettant d'inclure des caractères non-ASCII

"""
Created on Tue May  3 09:35:04 2022
https://askcodez.com/opencv-a-laide-de-cv2-approxpolydp-correctement.html
# URL de la source d'inspiration ou de référence pour le script
@author: BEBO
# Informations de création du script et auteur
"""

import numpy as np
# Importe la bibliothèque NumPy pour les opérations numériques sur des tableaux multidimensionnels
import cv2
# Importe la bibliothèque OpenCV pour la manipulation d'images et la vision par ordinateur
import matplotlib.pyplot as plt
# Importe la bibliothèque matplotlib pour la création de graphiques
import time
# Importe la bibliothèque time pour les opérations de gestion du temps

# Charger l'image et la réduire - l'image est massive

img = cv2.imread('UK1.png')
# Charge l'image depuis le fichier 'UK1.png'

img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
# Redimensionne l'image en réduisant ses dimensions par un facteur de 0.3 en utilisant l'interpolation bicubique

# Obtenir une toile vierge pour dessiner le contour et convertir l'image en niveaux de gris

canvas = np.zeros(img.shape, np.uint8)
# Crée une image vide (noire) de la même taille que l'image d'origine pour dessiner les contours
img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Convertit l'image en niveaux de gris

# Filtrer les petites lignes entre les comtés

kernel = np.ones((5, 5), np.float32) / 25
# Crée un noyau de filtre de moyenne de taille 5x5
img2gray = cv2.filter2D(img2gray, -1, kernel)
# Applique le filtre de moyenne à l'image en niveaux de gris

# Afficher l'image originale et l'image filtrée côte à côte
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img2gray), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
# Affiche les deux images dans un graphique avec deux sous-graphiques
time.sleep(3)
# Fait une pause de 3 secondes pour permettre à l'utilisateur de voir le graphique

# Seuiler l'image et extraire les contours
# Le seuillage est une technique dans OpenCV,
# qui assigne des valeurs de pixels en relation avec la valeur de seuil fournie.
# Dans le seuillage, chaque valeur de pixel est comparée avec la valeur de seuil.
# Si la valeur du pixel est inférieure au seuil, elle est définie à 0,
# sinon, elle est définie à une valeur maximale (généralement 255)
# L'image d'entrée doit être en niveaux de gris.

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# Applique un seuillage binaire à l'image
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# Applique un seuillage binaire inversé à l'image
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# Applique un seuillage tronqué à l'image
ret, thresh4 = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
# Applique un seuillage à zéro à l'image
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
# Applique un seuillage inversé à zéro à l'image
ret, thresh6 = cv2.threshold(img2gray, 250, 255, cv2.THRESH_BINARY_INV)
# Applique un seuillage binaire inversé à l'image en niveaux de gris (code d'origine)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV', 'Code Origine']
# Liste des titres pour les sous-graphiques
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6]
# Liste des images pour les sous-graphiques

for i in range(7):
    plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    # Crée un sous-graphique pour chaque image
    plt.title(titles[i])
    # Ajoute un titre au sous-graphique
    plt.xticks([]), plt.yticks([])
    # Supprime les graduations des axes x et y
plt.show()
# Affiche les sous-graphiques
time.sleep(3)
# Fait une pause de 3 secondes pour permettre à l'utilisateur de voir les graphiques

# Points du contour (x, y)
contours, hierarchy = cv2.findContours(thresh6, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# Trouve les contours dans l'image seuillée (thresh6)

# Trouver l'île principale (plus grande aire)
cnt = contours[0]
# Initialise le contour principal avec le premier contour trouvé
max_area = cv2.contourArea(cnt)
# Calcule l'aire du contour initial

for cont in contours:
    print('cont \n', cont)
    # Imprime les points de chaque contour
    if cv2.contourArea(cont) > max_area:
        cnt = cont
        max_area = cv2.contourArea(cont)
        # Met à jour le contour principal si un contour avec une aire plus grande est trouvé

# Définir l'approximation du contour principal et son enveloppe convexe
perimeter = cv2.arcLength(cnt, True)
# Calcule la longueur de l'arc (périmètre) du contour principal
epsilon = 0.01 * cv2.arcLength(cnt, True)
# Calcule epsilon, la distance maximale entre le contour approximatif et le contour réel
approx = cv2.approxPolyDP(cnt, epsilon, True)
# Approxime le contour principal
hull = cv2.convexHull(cnt)
# Calcule l'enveloppe convexe du contour principal

cv2.isContourConvex(cnt)
# Vérifie si le contour principal est convexe

# Dessiner les contours sur la toile
cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)
# Dessine le contour principal en vert sur la toile
cv2.drawContours(canvas, approx, -1, (0, 0, 255), 3)
# Dessine l'approximation du contour en rouge sur la toile
cv2.drawContours(canvas, hull, -1, (0, 0, 255), 3)
# Dessine l'enveloppe convexe en rouge sur la toile (ne montre que quelques points)
cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 3)
# Dessine l'approximation du contour en rouge sur la toile

cv2.imshow("Contour", canvas)
# Affiche l'image avec les contours dessinés
k = cv2.waitKey(0)
# Attend une touche pressée pour fermer la fenêtre affichée
