#!/usr/bin/env python

import cv2 as cv
import sys
import numpy as np
from PIL import Image

image_path = sys.argv[1]
image = cv.imread(image_path)
image_one = cv.cvtColor(image, cv.COLOR_BGR2XYZ)
image_two = cv.cvtColor(image_one, cv.COLOR_BGR2GRAY)

threshold = 128
num_of_rows = len(image)
num_of_cols = len(image[0])

cv.imwrite('images/temp_XYZgray.jpg', image_two)

ret,thresh = cv.threshold(image_two,127,255,0)
im2, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#image_three = cv.drawContours(image, contours, -1, (0,255,0), 3)
print type(contours)
print len(image)

idx = contours[0][0]
mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
cv.drawContours(mask, contours, -1, (255,255,255), -1) # Draw filled contour in mask
out = np.zeros_like(image) # Extract out the object and place into output image
out[mask == 255] = image[mask == 255]
cv.imwrite('images/temp_contours1.jpg',out)
# Show the output image
cv.imshow('Output', out)
cv.waitKey(0)
#cv.destroyAllWindows()

image_segmented = np.zeros((num_of_rows, num_of_cols, 1), np.uint8)
image_four = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
for ii in range(len(image_four)):
	for jj in range (len(image_four[0])):
		true_value = image_two[ii][jj] & image_four[ii][jj]
		if abs(true_value) != 0:
			image_segmented[ii][jj] = image_two[ii][jj]
			

cv.imwrite('images/image_segmented.jpg',image_segmented)
cv.imwrite('images/image_four.jpg', image_four)
#cv.imshow('im2', image_three)
#cv.waitKey()
#exit()
