#Drawing shapes and texts on images using opencv

import cv2
import numpy as np

#read the image
img=cv2.imread('iu.jpeg',cv2.IMREAD_COLOR)




#=======================================================
#                 Draw a line
#=======================================================

#line()
#Use
'''draw a line'''
#Parameters:
'''
(1)image 
(2)starting point
(3)ending point 
(4)color (b,g,r) 
(5)line width in px
'''
cv2.line(img,(0,0),(150,150),(255,255,255),15)



#=======================================================
#                 Draw a rectangle
#=======================================================

#recatngle()
#Use
'''draw a rectangle'''
#Parameters:
'''
(1)image 
(2)top left co-ordinates 
(3)bottom right co-ordinates
(4)color in (b,g,r) 
(5)line width
 '''
cv2.rectangle(img,(15,25),(100,150),(0,250,250),5)


#=======================================================
#                 Draw a circle
#=======================================================

#circle()
#Use
'''draw a circle'''
#Parameters 
'''
(1)image 
(2)centre 
(3)radius 
(4) color (b,g,r)
(5) -1 specifies cicle with filled colors
'''
cv2.circle(img,(16,7),15,(250,250,0),-1)

#=======================================================
#                 Draw a polygon
#=======================================================

#polylines()
#Use:
'''draw a polygon'''
#Parameters 
'''
(1)image 
(2)points 
(3)bool to join first and last points 
(4)color
(5)width
'''
pts=np.array([[12,34],[73,56],[19,37],[75,89],[69,96]],np.int32)
cv2.polylines(img,[pts],True,(200,255,0),3)


#=======================================================
#                 Write text
#=======================================================

#putText()
#Use:
'''Write text'''
#Parameters:
'''
(1)image
(2)text
(3)starting co-ordinates 
(4)font
(5)font-size
(6)color
(7)width
(8)for better look Line_AA is required
'''
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV Tuts',(0,130),font,0.3,(0,200,255),5,cv2.LINE_AA)

#=======================================================
#                 Output Image to screen
#=======================================================
cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()