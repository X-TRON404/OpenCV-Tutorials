#Here we are going to use haar cascade to detect objects
#we are going to detect eyes and face in this code example

import cv2
import numpy as np
# ===========================================
#           CascadeClassifier
# ===========================================
'''CascadeClassifier takes a pre trained classifier as an argument'''

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)


while True:
    #get a frame from the video
    ret,img = cap.read()
    #convert the frame to grey
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ===========================================
#           detectMultiScale()
# ===========================================
#Parameters:
    '''
    (1)region of image
    '''
    faces = face_cascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
#to make rectangles where face is detected
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]

#to make rectangles where eye is detected
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xFF
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()


