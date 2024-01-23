import numpy as np
import cv2
from matplotlib import pyplot as plt

imgLeft = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
imgRight = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

imgLeft = cv2.resize(imgLeft,(231,310))
imgRight = cv2.resize(imgRight,(231,310))

print(imgLeft.shape, imgRight.shape)
print(imgLeft.dtype, imgRight.dtype)

# Create a StereoSGBM object
stereo = cv2.StereoSGBM_create(
    numDisparities=2,
    blockSize = 15
)
disparity = stereo.compute(imgLeft, imgRight)
plt.imshow(disparity, 'gray')
plt.show()