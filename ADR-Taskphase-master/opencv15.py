#Here we are going to separate foreground from background based on the fact that the moving objects are foreground

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fgbg= cv2.createBackgroundSubtractorMOG2()
#=======================================================
#                   createBackgroundSubtractorMOG2()
#=======================================================
#Parameters:
'''
(3)detectShadows bool :to detect shadows True by default
'''

while True:
    ret,frame=cap.read()
    
#to apply oneach frame of the video
    fgmask =fgbg.apply(frame)
   
    cv2.imshow('frame',frame)
    cv2.imshow('fgmask',fgmask)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
