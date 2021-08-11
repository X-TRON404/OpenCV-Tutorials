import cv2 
import numpy as np
import matplotlib.pyplot as plt



#imread()
#Use:
'''To read the images'''
#Parameters
'''(1)the image file'''
'''(2)the format of the image bgr,grayscale,etc'''

#Options for the 2nd parameter:
'''
IMREAD_GRAYSCALE is the default value for .imread method / use 0

IMREAD_COLOR=1 coloured image in 'bgr' format / use 1

IMREAD_UNCHANGED=-1 this doesnt change anything / use -1
 '''

#convert colored to grayscale:
img=cv2.imread('iu.jpeg',cv2.IMREAD_GRAYSCALE)




'''greyscale images are easy to work with because there you just have one colour'''

'''coloured images in matplotlib are 'rgb' while in opencv they are 'bgr' '''

'''The 'aplha' factor takes care of the opacity of the images'''




#imshow():
#Use:
'''To output the image'''
#Parameters:
''' (1)'image' is the title to that image '''
''' (2) The image which we want to output'''



cv2.imshow('image',img)


k = cv2.waitKey(0) & 0xFF

# wait for ESC key to exit
if k == 27:         
    cv2.destroyAllWindows()


# wait for 's' key to save and exit
elif k == ord('s'): 
    #im.write()
    #Use:
    '''Used to save the image'''
    #Parameters:
    ''' (1)to save as name
        (2)image to save'''

    cv2.imwrite('iu.jpeg',img)
    cv2.destroyAllWindows()


