#template matching
#to find a particular object from the image,given a template

import cv2
import numpy as np

img_rgb = cv2.imread('opencv-template-matching-python-tutorial.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('opencv-template-for-matching.jpg',0)

#dimensions of template

#shape[::-1] reverses the order in which elemets are stored in the template.shape tuple 
w, h = template.shape[::-1]
#=======================================================
#                 match template
#=======================================================
#matchTemplate()
#Algorithm:
'''(1)It simply slides the template image over the input image (as in 2D convolution) and compares the template and patch of input image under the template image.'''
'''(2)It returns a grayscale image, where each pixel denotes how much does the neighbourhood of that pixel match with template.'''
#Use:
'''Template Matching is a method for searching and finding the location of a template image in a larger image.'''
'''If input image is of size (WxH) and template image is of size (wxh), 
output image will have a size of (W-w+1, H-h+1).'''
#Parameters:
'''
(1)Grayscale image (as it uses 2D convolution)
(2)template
(3)Comparision method (template match mode)
'''

#res is a greyscale image 
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8

#threshold defines % match
#here,in this case match should be at least 80%
#if threshold is reduced too much then false positive results may occur

#Function: 'where()' 
# 'where' function returns an array or a tuple of arrays which satisfy the condition


#loc returns a tuple which contains two arrays of points which satisy the threshold condition.
'''first array contains all the points in y direction which satisfy the threshold condition''' 
'''second array contains all the points in y direction which satisfy the threshold condition'''
loc = np.where( res >= threshold)



#Function:zip()
#zip method elementwise pairs up the input tuples  
'''The only condition is that the number of arguments a function
takes should be equal to the number of elements in the list or a tuple'''


# *loc[::-1] reveses the index of the elements stored in the loc matrix 
# '*' is used for unpacking lists and tuples in a function.



#here given loop joins all the points which satisfy the condition threshold 0.8 and creates a rectangle
#here pt is a tuple with (x,y) of the points which satisfy the threshold 0.8 condition  
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 1)

cv2.imshow('Detected',img_rgb)
'''as there are multiple points which are very near and satisfy the threshold condition,
you may see a thick bordered rectangle around the part of the image which resembles the
template even when you kept boder equal to 1 in rectangle'''

cv2.waitKey(0)
cv2.destroyAllWindows()