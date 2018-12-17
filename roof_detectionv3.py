#!/usr/bin/env python

import cv2
import sys
import numpy as np
from PIL import Image

image_path = sys.argv[1]
original_image = cv2.imread(image_path)
# XYZ_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2XYZ)
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Gobol filtering
kernel_size = 9
sigma = 3
theta = [30, 60, 90, 120, 150, 180]
lambd = 4
gamma = 0.04
psi = 0
combined_image = np.zeros_like(np.float64(gray_image))

for ii in range(6):
	gobor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta[ii], lambd, gamma, psi, ktype = cv2.CV_64F)
	im_gobor_filtered = cv2.filter2D(gray_image, cv2.CV_64F, gobor_kernel)
	image_index = "images/" + `ii` + ".jpg"
	cv2.imwrite(image_index, im_gobor_filtered)
	combined_image +=  im_gobor_filtered

print np.min(combined_image)
print np.max(combined_image)
combined_image = np.uint8((combined_image-np.min(combined_image))/(np.max(combined_image)-np.min(combined_image))*255)
#combined_image = np.abs(combined_image)
#combined_image = np.uint8((combined_image)/(np.max(combined_image))*255)

#combined_image += gray_image

print np.min(combined_image)
print np.max(combined_image)

cv2.imwrite('images/combined_image.jpg', combined_image)
	


