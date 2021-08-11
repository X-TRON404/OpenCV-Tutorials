#here we are going to filter a photo or a video for a specific color or range of colors 
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


    '''Similar to how we did thresholding for greyscale images 
    we can do that for coloured images using inRange()  '''
    '''Parameters (1)value (2)Upper_bound (3)Lower_bound'''


    #the mask will return a mask for the hsv values of the image which are lying between upper and lower bounds
    mask=cv2.inRange(hsv_img,lower_red,upper_red)


    '''the part of the image from the mask which has high pixel value for corresponding 
    pixels in original frame will give us the segmented color from the image'''

    result=cv2.bitwise_and(frame,frame,mask=mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('result',result)


#here as we are using frames from the video we have to 'break' from th loop in order to stop the incoming video
#is we dont use 'break'then it will just stop the recording for that particular frame and will continue with next frame
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()