#gradient and edge detection
#laplacian edge detector

#canny gives edges based on region Canny(image,x,y)

#What is an edge?
'''Sudden change in the intensity of pixels'''


import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()
#=======================================================
#                 Laplacian edge detection
#=======================================================
#technique:
    '''
    (1)The Laplacian L(x,y) of an image with pixel intensity values I(x,y) is given by double differentiating the I with x & y and adding it

                L(x,y)=Ixx+Iyy

    (2)Since input images is a set of discrete pixels,we use convolutional kernel that approximates second derivatives in the definition of the laplacian

    The most common two kernels are:

        [[0,-1,0],          [[-1,-1,-1],
        [-1,4,-1],   &      [-1,8,-1],   
        [0,-1,0]]           [-1,-1,-1]]
        
    '''

#Laplacian()
#Use:
    '''Edge detection in all directions'''
#Parameters:
    '''  
    (1)image
    (2)Output matrix datatype
    '''


    laplacian = cv2.Laplacian(frame,cv2.CV_64F)


#Note:
    '''Use gaussian blur before using sobel for better performance
    (get rid of noise hence better edge detection)'''
#=======================================================
#                 Sobel edge detection
#=======================================================
# Sobel works only with grayscale images  
#for detecting the edges the computer usually looks for sharp changes in the pixel intensity
#To do that it calculates the derivatives

    '''
        kernel  for gradient in x direction     [[-1,0,1],     
                                                [-2,0,2],
                                                [-1,0,1]]
    
        kernel  for gradient in y direction   
                                                [[-1,-2,-1],     
                                                [0,0,0],
                                                [1,2,1]]
        Gradient=kernel*Image

        total gradient= (Gx^2+Gy^2)^1/2

        we can find the orientation of these edges with atan(Gx/Gy) 
    '''

#sobel()
#Use:
    '''Edge detection in grayscale images'''
#Technique:
    '''the sobel operator uses kernel convolution technique and it returns a high response
    where the intensity changes rapidly and low response where it doesnt.'''
#Parameters:
    '''
    (1)image
    (2)desired depth of the output image
    (3)The order of derivative in x-direction
    (4)The order of derivative in y-direction
    (5)kernel size
    '''

    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)

#=======================================================
#                 Canny edge detection
#=======================================================
#Algorithm:
    '''The output of the sobel edge detector is the input of the canny edge detector'''

    '''(1)canny thins out the edges from the sobel edge detector:
    (a)from sobel we already have edges and edge angles.
    (b)with these edge angles and edges it finds the local maxima along the edge direction.


    (2)then it uses histeresis thresholding:
    (a)hystersis thresholding uses two thresholds.
    (b)If the value is lower than the smaller threshold then it is scrapped
    (c)If it is greater than the bigger thrshold then it is accepted.
    (d)Any value in between the two thresholds is only accepted if these pixels are 
    connected to the pixels above the bigger threshold.
    (e)This is determined by pixel traversing
    '''
#Canny()
#Use:
    '''(1)Thinning and sorting out significant edges'''
#Parameters:
    '''
    (1)image
    (2)lower threshold
    (3)higher threshold
    '''

    canny = cv2.Canny(frame,100,200)

    cv2.imshow('Original',frame)
    cv2.imshow('canny',canny)
    cv2.imshow('laplacian',laplacian)
    cv2.imshow('sobelx',sobelx)
    cv2.imshow('sobely',sobely)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()