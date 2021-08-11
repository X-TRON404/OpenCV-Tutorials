#Here we are going to work with Gaussian adaptive thresholding
#adaptive thresholding is used when we want to apply thresholding for a small region
#when lighting conditions in image change from point to point we use adaptive thresholding
#hence we get different thrsholding values for different regions in the same image
import cv2
import numpy as np


img=cv2.imread('iu.jpeg')


#here you can see that images from thresh1 and thresh2 are not good as they are thresholding globally
ret,thresh1=cv2.threshold(img,12,255,cv2.THRESH_BINARY)

greyscaled=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh2=cv2.threshold(img,12,255,cv2.THRESH_BINARY)


#=======================================================
#                 Adaptive thresholding
#=======================================================

#adaptiveThreshold()
#Use:
'''For applying thresholding to a smaller region instead of applying it to whole image'''
#Parameters:
'''
(1)img
(3)max_val
(4)adaptiveMethod :
(5) thresholdType: The type of thresholding to be applied.
(6) blockSize: Size of a pixel neighborhood that is used to calculate a threshold value.
(7) constant: A constant value that is subtracted from the mean or weighted 
sum of the neighbourhood pixels.
'''

#cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 
'''Threshold Value = (Gaussian-weighted sum of the neighbourhood values – constant value). 
The threshold value T(x,y) is weighted sum of the blocksize*blocksize  neighborhood minus C
where c is a constant
'''
gaus=cv2.adaptiveThreshold(greyscaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)

#=======================================================
#                 Output Images to screen
#=======================================================

cv2.imshow('original',img)

cv2.imshow('thresh1',thresh1)

cv2.imshow('thresh2',thresh2)

#image with gaussian adaptive thresholding gives good result as it is thresholded locally
cv2.imshow('gaussian',gaus)

cv2.waitKey(0)
cv2.destroyAllWindows()