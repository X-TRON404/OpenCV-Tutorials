#Image adding
#Here we  are going to add different images to each other 

import numpy as np
import cv2

img1=cv2.imread('3D-Matplotlib.png')
img2=cv2.imread('mainlogo.png')

#=======================================================
#                 Add two images
#=======================================================

#Directly add two images
#add_direct=img1+img2
#using openCV add function


#add()
#Use:
'''adds the bgr pixel values of the two images'''
#But the problem is, if the addition exceeds  255 it shows white colored pixel'''
#Parameters :
'''
(1)image1
(2)image2
'''
# add_func=cv2.add(img1,img2)



#=======================================================
#                 Weighted addition of two images
#=======================================================

#addWeighted()
#weighted addition is '''Preferred'''

#function:
'''g(x)=(1−α)f0(x)+αf1(x)+γ'''

#Use:
'''Here we assign weights to the images and then add them'''
#Parameters:
'''
(1)img1 
(2)weight* number of images 
(n)Gammma = bias to add
'''
alpha = 0.6
beta = 1-alpha
weighted=cv2.addWeighted(img1,alpha,img2,beta,0)



#=======================================================
#                 BACKGROUND REMOVAL
#=======================================================

#Unpacking the shape tuple
rows,cols,channels=img2.shape

'''Now we are defining our region of interest to be the top left corner 
which has the same dimensions as the img2'''
'''We will be adding our img2 here'''

roi=img1[0:rows,0:cols]
img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)



#=======================================================
#                 Thresholding
#=======================================================

#threshold() (global)
#Use:
'''Usually thresholding is used for separating background of the image from the foreground'''

#Technique:
'''Threshoding is a technique in computer vision in which if a pixel value is less than the
 threshold then it is converted to zero and if the threshold value is greater than the 
 threshold then it is converted to the maximum value(usually 255).'''

 #Parameters:
'''
(1)img
(2)min_threshold
(3)max_val
(4)thresholding technique'
'''

#The different Simple Thresholding Techniques are:
'''
cv2.THRESH_BINARY: If pixel intensity is greater than the set threshold, value set to 255,
 else set to 0 (black).

cv2.THRESH_BINARY_INV: Inverted or Opposite case of cv2.THRESH_BINARY.

cv.THRESH_TRUNC: If pixel intensity value is greater than threshold, it is truncated to the 
threshold. The pixel values are set to be the same as the threshold. All other values remain 
the same.

cv.THRESH_TOZERO: Pixel intensity is set to 0, for all the pixels intensity, 
less than the threshold value.

cv.THRESH_TOZERO_INV: Inverted or Opposite case of cv2.THRESH_TOZERO.'''


#We are more concerned about binary thresholding here,where we are converting colored images into black and white images


 
#=======================================================
#                      Mask
#=======================================================
'''masks are binary images which are  balck and white'''

#here we get our mask where we change bg to black and fg to white
ret,mask=cv2.threshold(img2gray,220,255,cv2.THRESH_BINARY_INV)

#now we invert the color of the mask
mask_invis=cv2.bitwise_not(mask)


#=======================================================
#         Bitwise operations in image processing
#=======================================================
'''bitwise_and returns 1 if both the images have 1 pixel value for that specific pixel else returns 0'''
'''bitwise_or returns 1 if any one of the images have pixel value 1 for that particualar pixel'''
'''bitwise_not turns 1's into 0's and 0's into 1's'''



#The operation of "And" will be performed only if mask[i] doesn't equal zero, else the the result of and operation will be zero
img1_bg=cv2.bitwise_and(roi,roi,mask=mask_invis)
img2_fg=cv2.bitwise_and(img2,img2,mask=mask)



#up until here we have converted img2 to have the background of img1
destin=cv2.add(img1_bg,img2_fg)

#now we will add the destin in the foreground of img1

img1[0:rows,0:cols]=destin


#=======================================================
#                 Output Images to screen
#=======================================================
cv2.imshow('image_with_changed_foreground',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
