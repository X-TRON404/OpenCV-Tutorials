#image operations 
'''Here we are going to look at different operations like
 copying a part of image modifying parts of image,etc'''


import numpy as np
import cv2

img=cv2.imread('iu.jpeg',cv2.IMREAD_COLOR)


#=======================================================
#                Change pixel value
#=======================================================

'''make pixel at position (55,55) white'''
img[55,55]=[255,255,255]
#store that pixel value in a variable for our reference
px=img[55,55]

#=======================================================
#                 Region of interest
#=======================================================
'''ROI'''
#Region Of Image is basically a subimage within the image
'''here we are making pixels from (100 to 150)X and (100,150)Y white'''
img[100:150,100:150]=[255,255,255]

#=======================================================
#                 Copy a part of image
#=======================================================
'''copying a part of image and pasting it within the image with ROI'''
part_to_copy=img[170:195,300:330]
'''make sure that region where the part is copied has the same size as the part to be copied'''
img[0:25,0:30]=part_to_copy

#=======================================================
#                 Output Image to screen
#=======================================================
cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
