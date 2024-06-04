# -*- coding: utf-8 -*-
"""
Created on Tue May  3 09:35:04 2022
https://askcodez.com/opencv-a-laide-de-cv2-approxpolydp-correctement.html
@author: BEBO
"""

import numpy as np
import cv2 # voir la version cv2 n'est pas une release
import matplotlib.pyplot as plt
import time

# load image and shrink - it's massive

img = cv2.imread('F:\Cours PYTHON/UK1.png')

img = cv2.resize(img, None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)

# get a blank canvas for drawing contour on and convert img to grayscale

canvas = np.zeros(img.shape, np.uint8)
img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# filter out small lines between counties

kernel = np.ones((5,5),np.float32)/25
img2gray = cv2.filter2D(img2gray,-1,kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img2gray),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
time.sleep(3)

# threshold the image and extract contours
# Thresholding is a technique in OpenCV,
# which is the assignment of pixel values in relation to the threshold value provided.
# In thresholding, each pixel value is compared with the threshold value. 
#If the pixel value is smaller than the threshold, it is set to 0, 
#otherwise, it is set to a maximum value (generally 255)
#input Image array must be in Grayscale. 

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,100,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
ret,thresh6 = cv2.threshold(img2gray,250,255,cv2.THRESH_BINARY_INV) # code d'origine
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV','Code Origine']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6]
for i in range(7):
    plt.subplot(3,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
time.sleep(3)


# points du contour (x,y)
contours,hierarchy = cv2.findContours(thresh6, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# find the main island (biggest area)
# tableau des points du contou
cnt = contours[0]
max_area = cv2.contourArea(cnt)
for cont in contours:
    print('cont \n',cont)
    if cv2.contourArea(cont) > max_area:
        cnt = cont
        max_area = cv2.contourArea(cont)
# define main island contour approx. and hull
perimeter = cv2.arcLength(cnt,True)
epsilon = 0.01*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
hull = cv2.convexHull(cnt)

cv2.isContourConvex(cnt)

cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)
cv2.drawContours(canvas, approx, -1, (0, 0, 255), 3)
cv2.drawContours(canvas, hull, -1, (0, 0, 255), 3) # only displays a few points as well.
cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 3)
cv2.imshow("Contour", canvas)
k = cv2.waitKey(0)

    

