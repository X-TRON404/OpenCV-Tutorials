#corner detection in opencv
#here we are going to learn how to detect corners in the images

#What are corners?
'''Corners are points where two edges meet'''
#why corners are considerd good features for patch mapping?
'''Consider a window of (n*n) sliding over (m*m) image where m>n'''
'''
3 Scenarios:
(1)Now if you slide this window where there is neither a corner nor a edge there wont be any
change in the pixel values of the original window and the shifted window.

(2)When there is an edge there wont be any change in the pixel intensity if you move the
window along the direction of the edge ,but there will be a change in the pixel intensity
of the shifted and original window. 

(3)In corners the pixel intensities of the original and shifted windows changes in all directions
Hence,shifting a window in any direction will bring forward large differences between two patches of the image.
'''


import cv2
import numpy as np

img = cv2.imread('opencv-corner-detection-sample.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#we need a float valued image for harris detector
#hence convert the pixel values to float
gray = np.float32(gray)

#=========================================================================================
#                            Harris corner detector
#=========================================================================================

#Algorithm:
'''
(1)Our window is a rectangular window which assigns weights to the pixels of the image inside the window.

(2)Window function w(x,y) acts as a window.

(3)It finds the sum of squared difference in intensity for displacement of (u,v) in all directions.
(4)                       E(u,v)= sigma{w(x,y)*[I(x+u,y+v)-I(x,y)]^2}   where,

w(x,y) = window function
I(x+u,y+v) = shifted intensity
I(x,y) = original intensity

(5)We have to maximize this function E(u,v) for corner detection.
That means, we have to maximize the second order term.

(6)Applying Taylor Expansion to above equation to simplify the above equation because it is computationally heavy.
we get
E(u,v)=[u,v]*M*[u,v]^T

where M = sigma{w(x,y)*[IxIx IxIy
.                       IxIy IyIy]}
 
(7)Ix and Iy are image derivatives in x and y directions respectively
(8)Now we calculate the eigen values for M

(9)Now after this we calculate a score (R) which determines whether a fiven patch is a window or not

(10) R = det(M) - k(trace(M)) 

(11)the det(M)=l1*l2 & trace(M)=l1+l2 where l1 and l2 are the eigen values of M

(12)When R=0    flat region
.   When R<0    edge
.   When R>0    corner'''




#=========================================================================================
#                             goodFeaturesToTrack
#=========================================================================================
#Use:
'''To determine the corners in an image'''
#Parameters:
'''
(1)image

(2)maxCorners-maximum number corners to be detected

(3)qualityLevel – Parameter characterizing the minimal accepted quality of image corners

(4)minDistance – Minimum possible Euclidean distance between the returned corners

(5)blockSize – Size of an average block for computing a derivative covariation matrix 
over each pixel neighborhood.

(6)useHarrisDetector – Parameter indicating whether to use a Harris detector or not

(7)k – Free parameter of the Harris detector.
'''

corners = cv2.goodFeaturesToTrack(gray, 10, 0.01, 10)


#as pixel locations (x,y) are integers we have to convert the float values returned from the function to int  
corners = np.int0(corners)


#Function: np.ravel()
#Use:
#array.ravel() works in the same way as array.reshape(-1)
#both of them flatten the given array into 1-D array.

for corner in corners:
    #flattening the n dimensional corners array to 1-D array
    x,y = corner.ravel()

    #wherever there is a corner make a circle
    cv2.circle(img,(x,y),3,255,-1)
    
cv2.imshow('Corner',img)
cv2.waitKey(0)
cv2.destroyAllWindows()