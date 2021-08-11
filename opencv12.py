#to segment foreground from the background  
#here we are going to use grabcut to segment foreground from background


import cv2
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread('iu.jpeg')



#mask with (width,height) = [x,y] with all zeros
mask=np.zeros(img.shape[0:2],np.uint8)
cv2.imshow('mask',mask)

# bgdModel dimensions=[1*65] with all zeros(1*65 matrix is required by the grabcut algorithm)
bgdModel = np.zeros((1,65),np.float64)

# fgdModel dimensions=[1*65] with all zeros(1*65 matrix is required by the grabcut algorithm)
fgdModel = np.zeros((1,65),np.float64)

#=======================================================
#                   GrabCut
#=======================================================
#Algorithm:
'''
(1)Initially user draws a rectangle around the foreground region.
(foreground should be comletely inside the rectangle)

(2)Everything outside the rectangle is definitely a background.

(3)Computer does initial labeling based on the data we give.
It labels foreground and background pixels.

(4)Now a Gaussian Mixture Model(GMM) is used to model the foreground and background.

(5)Depending on the data we gave, GMM learns and create new pixel distribution. 
That is, the unknown pixels are labelled either probable foreground or probable background 
depending on its relation with the other hard-labelled pixels in terms of color statistics

(6)Anything inside the rectangle can be a foreground,probable foreground,background,
probable background

(7)We make a graph of pixels with each pixel as a node and assign two additional nodes source
and sink.Every foreground pixel is connected to the source node and every
 background pixel is connected to the sink node.

(8)The weights of edges connecting pixels to source/sink node are defined by the probability
 of that pixel being in foreground/background.
The weights of the edges connecting pixels in the graph are determined by the pixel similarity.
greater the difference in pixel values lesser the weight.

(9)Then a MinCut algorithm is applied to cut through all the edges with low weights and it gives
a segmented image. 
'''
#Use:
'''to separate foreground from background'''
#Parameters:
'''
(1)img - Input image

(2)mask - It is a mask image where we specify which areas are background, 
foreground or probable background/foreground etc. It is done by the following flags, 
cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD, or simply pass 0,1,2,3 to image.

(3)rect - It is the coordinates of a rectangle which includes the foreground object
 in the format (x,y,w,h)

(4)bdgModel, fgdModel - These are arrays used by the algorithm internally. 
You just create two np.float64 type zero arrays of size (1,65).

(5)iterCount - Number of iterations the algorithm should run.

(6)mode - It should be cv.GC_INIT_WITH_RECT or cv.GC_INIT_WITH_MASK or
 combined which decides whether we are drawing rectangle or final touchup strokes.
'''

#everything outside the rectangle is definitely background
rect=(90,10,381,343)#(x,y,w,h)

#apply grabcut
#grabcut will make changes to the mask image that we have described above
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

#where there is probable background(2) and definite background(0) assign 0 else assign 1 in the mask
mask2=np.where((mask==2) | (mask==1) ,0,1).astype('uint8')

#to get the segmented image multiply original image with the mask2
img=img*mask2[:,:,np.newaxis]

#matplotlib displays data as rgb but cv2 reads it as bgr so there may be a color difference in the output image
plt.imshow(img)
plt.colorbar()
plt.show()


