# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:41:26 2022

@author: BEBO
"""

import numpy as np
import cv2                            # voir la version cv2 n'est pas une release opencv
import matplotlib.pyplot as plt
import time

# load image and shrink

# img = cv2.imread('./chelsea.png')
# cv2.imshow("chelsea",img)

img = cv2.imread('./chien.jpg')
cv2.imshow("chien",img)

img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

# get a blank canvas for drawing contour on and convert img to grayscale

canvas = np.zeros(img.shape, np.uint8)  # img.shape  dimensions HxL

img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# filter out small lines between counties

kernel = np.ones((5, 5), np.float32)/25  # initialise  array a 1 5 col 5 lignes

img2gray = cv2.filter2D(img2gray, -1, kernel)

plt.subplot(121), plt.imshow(img), plt.title('Original')

plt.xticks([]), plt.yticks([])  # marquage

plt.subplot(122), plt.imshow(img2gray), plt.title('Averaging')

plt.xticks([]), plt.yticks([])

plt.show()
time.sleep(1)

# Seuil l'image et extrait les contours
# Le seuillage est une technique en OpenCV,
# qui consiste à attribuer des valeurs de pixels en relation avec la valeur de seuil fournie.
# Lors du seuillage, chaque valeur de pixel est comparée avec la valeur de seuil.
# Si la valeur du pixel est inférieure au seuil, elle est fixée à 0,
# sinon, elle est fixée à une valeur maximale (généralement 255).
# L'image d'entrée doit être en niveaux de gris.

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # limite
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
ret, thresh6 = cv2.threshold(
    img2gray, 127, 255, cv2.THRESH_BINARY_INV)  # code d'origine

titles = ['Original Image', 'BINARY', 'BINARY_INV',
          'TRUNC', 'TOZERO', 'TOZERO_INV', 'Code Origine']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6]

for i in range(7):
    plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
time.sleep(1)


#im2 = cv2.findContours(thresh6, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#im2,contours,hierarchy= cv2.findContours(thresh6, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

contours, hierarchy = cv2.findContours(
    thresh6, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# find the main picture (biggest area)

cnt = contours[0]
max_area = cv2.contourArea(cnt)
for cont in contours:
    print("Contour \n", cont)
    if cv2.contourArea(cont) > max_area:
        cnt = cont
        print('\n cnt1 \n')
        max_area = cv2.contourArea(cont)

# define main  contour approx. and hull

print('\n cnt2 \n')
perimeter = cv2.arcLength(cnt, True)
epsilon = 0.01*cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
hull = cv2.convexHull(cnt)
cv2.isContourConvex(cnt)
cv2.drawContours(img, cnt, -1, (127, 255, 0), 3)
cv2.drawContours(canvas, approx, -1, (127, 0, 255), 3)
# only displays a few points as well.
cv2.drawContours(canvas, hull, -1, (0, 0, 255), 3)
cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 3)


#cv2.imshow("Contour", canvas)

cv2.imshow("Contour", img2gray)

plt.imshow(img2gray)


k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
