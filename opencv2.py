#video analysis


import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while True:
    retrn,frame=cap.read()
    #frame = current frame
    #retrn = got the frame(True) or not(False)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow('gray',gray)
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

