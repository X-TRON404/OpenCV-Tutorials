'''for matching images which differ in lighting and have a diferent
orientation we cant use our old template matching method.'''
#here we are going to use feature matching for all those images which differ in lighting,orientation,etc'''


#Keypoint Features:
'''Small region in an image that is particularly distinctive'''
#Feature Descriptor:
'''Feature descriptors store unique information about feature into a series of numbers (arrays) 
and act as a numerical fingerprint of the feature'''
'''Ideally these feature descriptors should not change with any transformations in the image like rotation,etc'''

#=========================================================================================
#                            ORB  (Oriented FAST and Rotated BRIEF)
#=========================================================================================

#ORB is a combination of two algorithms FAST and BRIEF with some added features to improve performance.

#(1)FAST (Features from Accelerated and Segments Test):
#Use:
'''Keypoint detection'''
#Algorithm:
'''
(1)Given a pixel p in an array fast compares the brightness of p to surrounding 16 pixels that are in a small circle around p. 

(2)Pixels in the circle is then sorted into three classes (lighter than p, darker than p or similar to p). 

(3)If more than 8 pixels are darker or brighter than p than it is selected as a keypoint. 
So keypoints found by fast gives us information of the location of determining edges in an image.

'''
#BREIF(Binary Robust Independent Elementary Feature): 
#Use:
'''Feauture descriptor'''
#Algorithm:
'''Brief takes all keypoints found by the fast algorithm and converts it into a binary 
feature vector so that together they can represent an object'''

'''Binary features vector also know as binary feature descriptor is a feature vector that 
 contains only 1 and 0. 
In brief, each keypoint is described by a feature vector which is 128–512 bits string.'''

'''Brief perfoems poorly with rotation'''

'''ORB rotates the Brief according to theorientation of keypoints as breif does it poorly.''' 

#class ORB_create()
#Use:
'''to create orb object'''
#ORB Algorithm:
'''
(1)Take the query image and convert it to grayscale.
(2)Now Initialize the ORB detector and detect the keypoints in query image and scene.
(3)Compute the descriptors (binary feature vector) belonging to both the images.
(4)Match the keypoints using Brute Force Matcher.
(5)Show the matched images.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('opencv-feature-matching-template.jpg',0)
img2 = cv2.imread('opencv-feature-matching-image.jpg',0)

#create an orb_detector object
orb=cv2.ORB_create()

#find keypoints and descriptors with the orb detector and store in lists
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

#=========================================================================================
#                                 BFMatcher
#=========================================================================================
#class BFMatcher() 
#Use:
'''Match features between query and train image'''
#Algorithm:
'''Brute Force Matcher takes descriptor of one of the features in query image and matches it with all other features in the train image and the feature discriptor in train image which mathes the closest to the feature descriptor of the query image is returned'''
#Parameters:
'''
(1)Distance measurement technique. NORM_HAMMING should be used for binary string based descriptors like ORB, BRISK and BRIEF
(2)Second is a boolean variable, crossCheck which is false by default.
'''
#creating BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

'''Once BFMatcher object is created, two important methods are BFMatcher.match() and BFMatcher.knnMatch(). 
(1)BFMatcher.match() returns the best match. 
(2)BFMatcher.knnMatch() returns k best matches where k is specified by the user. 
It may be useful when we need to do additional work on that.'''

#match takes descriptors of query image and train image as the arguments and returns a list
matches = bf.match(des1,des2)


#lambda is used in python for annonymous functions
#here you can specify your own functions using lambda and pass them as an argument to key
#the objects inside the iterable will be sorted according to the lambda function 

#here key=lambda specifies to calculate the distance between the descriptors in the match list and sort them
matches = sorted(matches, key = lambda x:x.distance)


#to draw match lines between query image and train image
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
plt.imshow(img3)
plt.show()