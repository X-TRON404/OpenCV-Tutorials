#Using morphological transformations to remove noise from the image
#Erosion and Dilation 
#Opening for false positives and Closing for false negatives
#top hat and blackhat

#erosion:
'''(1)Erosion has a slider.It slides on the entire image and checks if all the pixels in the image have the same color or not.
(2)If there is one pixel that is not the same as the remaining pixels, we replace that pixel to have the same color as the others'''

#Dilation:
'''(1)Dilation has a slider.It slides on the entire image and checks if all the pixels in the image have the same color or not.
(2)If there is one pixel(p1) that is not the same as the remaining pixels, we replace all the pixel to have the same color as the p1'''


import cv2 
import numpy as np

cap=cv2.VideoCapture(0)
while True:

    ret,frame=cap.read()

    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_red = np.array([150,150,50])
    upper_red=np.array([180,255,150])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    kernel = np.ones((5,5),np.uint8)
    
    #remove false positives
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #remove false negatives
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Original',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('Opening',opening)
    cv2.imshow('Closing',closing)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
