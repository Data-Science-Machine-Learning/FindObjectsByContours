import cv2
import numpy as np

img = cv2.imread('input_file_find_objects.png') # reading image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting into gray scale


_, thresh = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY_INV) # doing thresholding on image

kernal = np.ones((2,2), np.uint8)
dilation = cv2.dilate(thresh, kernal, iterations=2) # doing dilation process, removing black distortion by kernal 2x2

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # finding contour shapes
#cv2.drawContours(dilation, contours, 1, (0, 255, 0), 1)

objects = str(len(contours)) # getting number of contours (objects found)

text = "Obj:"+str(objects)
cv2.putText(dilation, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.4,(240, 0, 159), 1) # printing number of objects on image

# showing original, threshold and dilation image
cv2.imshow('Original', img)
cv2.imshow('Thresh', thresh)
cv2.imshow('Dilation', dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()
