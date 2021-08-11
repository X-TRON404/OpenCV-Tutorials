#in the previous code we successfully extracted a specific colured object from the given image 
#but there was so much of noise in the image
#here we are going to look for the techniques to reduce the noise from the segmented images


import cv2
import numpy as np


#start capturing the video from default camera
cap=cv2.VideoCapture(0)


while True:
    ret,frame=cap.read()#returns a bool and a frame from video

    #converting the BGR color to hsv,since hsv colors are
    hsv_img=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #the upper and lower bounds for colour segmentation
    lower_red = np.array([150,150,50])
    upper_red=np.array([180,255,150])



#the mask will return a mask for the hsv values of the image which are lying between upper and lower bounds
#A pixel is set to 255 if it lies within the boundaries specified otherwise set to 0. This way it returns the thresholded image.


    mask=cv2.inRange(hsv_img,lower_red,upper_red)
    result=cv2.bitwise_and(frame,frame,mask=mask)




#All these techniques for removing noise use convolution with the help of kernels




#KERNEL:
#What is a kernel?:
    '''If our image is a big matrix then our kernel is a small matrix
    that slides from one end of the image to other and performs mathematical 
    operation (convolution) at each co-ordinate of the original image.
    An important part in the kernel is the center which is called anchor point.
    '''
#size:
    '''Kernels can be an arbitrary size of M x N pixels, provided
    that both M and N are odd positive integers'''




#convolution technique:
    '''
    1.Select an (x, y)-coordinate from the original image.

    2.Place the center of the kernel at this (x, y)-coordinate.

    3.Take the element-wise multiplication of the input image region and the kernel, 
    then sum up the values of these multiplication operations into a single value. 
    The sum of these multiplications is called the kernel output.

    4.Use the same (x, y)-coordinates from Step #1, but this time, store the 
    kernel output in the same (x, y)-location as the output image.'''

#the constant for kernel is determined by kernel_height*kernel_width
    kernel=np.ones((15,15),np.float32)/225


#Image sharpening:

#filter2D is used for image sharpening
    smoothed=cv2.filter2D(result,-1,kernel)


#Image Blurring (Image Smoothing):

#1.Gaussian blur :
#Use:
    '''Used for removing Gaussian noise'''

#technique:

#parameters:

    blur=cv2.GaussianBlur(result,(15,15),0)




#2.median blur :
#Use:
    '''This is highly effective in removing salt-and-pepper noise.'''
#technique:
    '''cv2.medianBlur() computes the median of all the pixels under the kernel 
    window and the central pixel is replaced with this median value.'''
#parameters:
    '''parameters (1)image (2)kernel size(box-kernel)'''
    median=cv2.medianBlur(result,15)



#3.bilateral blur
#Use:
    '''highly effective at noise removal while preserving edges'''
    '''cv2.bilateralFilter(), which was defined for, and is '''
#highly effective at noise removal while preserving edges

    bilateral=cv2.bilateralFilter(result,15,75,75)





    cv2.imshow('frame',frame)
    cv2.imshow('smoothed',smoothed)
    cv2.imshow('mask',mask)
    cv2.imshow('blur',blur)
    cv2.imshow('median',median)
    cv2.imshow('bilateral',bilateral)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
cap.release()

