#!/usr/bin/env python
# Spécifie l'interpréteur utilisé pour exécuter ce script Python

# -*- coding: utf-8 -*-
# Définit l'encodage des caractères utilisé dans le script pour supporter UTF-8

import cv2
# Importe la bibliothèque OpenCV pour la manipulation d'images et la vision par ordinateur

import numpy as np
# Importe la bibliothèque NumPy pour les opérations avancées sur les tableaux et les matrices

import imutils
# Importe la bibliothèque imutils pour simplifier certaines opérations sur les images

from webcolors import rgb_to_name, CSS3_HEX_TO_NAMES, hex_to_rgb
# Importe des fonctions et un dictionnaire pour manipuler les couleurs et leurs noms CSS3

from scipy.spatial import KDTree
# Importe KDTree de scipy.spatial pour effectuer des recherches de proximité rapide dans des données multidimensionnelles

def convert_rgb_to_names(rgb_tuple):
    # Déclare une fonction pour convertir une couleur RGB en son nom CSS3

    # Un dictionnaire contenant tous les codes hexadécimaux et leurs noms respectifs en CSS3
    css3_db = CSS3_HEX_TO_NAMES
    # Initialise des listes pour stocker les noms des couleurs et leurs valeurs RGB
    names = []
    rgb_values = []

    # Boucle sur chaque paire hex/nom dans la base de données CSS3
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    # Crée un arbre KDTree à partir des valeurs RGB pour une recherche rapide
    kdt_db = KDTree(rgb_values)
    
    # Requière la couleur la plus proche dans l'arbre KDTree
    distance, index = kdt_db.query(rgb_tuple)
    return names[index]
    # Retourne le nom de la couleur la plus proche trouvée

class ShapeDetector:
    # Classe pour détecter les formes géométriques

    def __init__(self):
        pass
        # Constructeur de la classe ShapeDetector (vide dans ce cas)

    def detect(self, c):
        # Méthode pour détecter la forme d'un contour donné

        # Initialise le nom de la forme comme "non identifiée"
        shape = "unidentified"
        # Calcule la longueur de l'arc du contour
        peri = cv2.arcLength(c, True)
        # Approxime le contour avec une précision proportionnelle à sa longueur
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # Détecte les formes selon le nombre de sommets approximés
        if len(approx) == 3:
            shape = "triangle"  # Triangles ont 3 sommets
        elif len(approx) == 4:
            # Carrés ou rectangles ont 4 sommets, il faut vérifier le rapport d'aspect
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # Si le rapport d'aspect est proche de 1, c'est un carré, sinon c'est un rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"  # Pentagones ont 5 sommets
        elif len(approx) == 6:
            shape = "hexagon"  # Hexagones ont 6 sommets
        elif len(approx) == 10 or len(approx) == 12:
            shape = "star"  # Les étoiles peuvent avoir 10 ou 12 sommets
        else:
            shape = "circle"  # Si aucun des cas précédents, on considère la forme comme un cercle

        # Retourne le nom de la forme détectée
        return shape

if __name__ == '__main__':
    # Bloc principal qui s'exécute si le script est exécuté directement

    # Charge l'image depuis le fichier et la redimensionne pour faciliter le traitement
    image = cv2.imread('python_shapes_detection_base.PNG')
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    
    # Convertit l'image redimensionnée en niveaux de gris, applique un flou gaussien et une binarisation
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    
    # Trouve les contours dans l'image binarisée et initialise le détecteur de formes
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()

    # Boucle sur chaque contour trouvé
    for c in cnts:
        # Calcule le moment du contour pour trouver son centre
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        
        # Détecte la forme à partir du contour
        shape = sd.detect(c)
        
        # Redimensionne le contour pour correspondre à l'image originale
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        
        # Crée un masque pour le contour détecté
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Convertit l'image en RGB et obtient la couleur moyenne du contour
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean = cv2.mean(imgRGB, mask=mask)[:3]
        named_color = convert_rgb_to_names(mean)
        
        # Calcule la couleur complémentaire pour afficher le texte
        mean2 = (255 - mean[0], 255 - mean[1], 255 - mean[2])
        
        # Prépare le texte à afficher (nom de la forme et couleur)
        objLbl = shape + " {}".format(named_color)
        textSize = cv2.getTextSize(objLbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.putText(image, objLbl, (int(cX - textSize[0] / 2), int(cY + textSize[1] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mean2, 2)
        
        # Affiche l'image avec les contours et le texte
        cv2.imshow("Image", image)
        # cv2.waitKey(0) # Attente d'une entrée clavier (désactivée pour des tests)
    cv2.waitKey(0) # Attente d'une entrée clavier pour fermer l'affichage
