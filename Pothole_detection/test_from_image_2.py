import cv2
import numpy as np
from matplotlib import pyplot as plt

bgr_img = cv2.imread('test.png') # read as it is

if bgr_img.shape[-1] == 3:           # color image
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
else:
    gray_img = bgr_img

img = cv2.medianBlur(gray_img, 5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

# Setting parameter values
t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold
  
# Applying the Canny Edge filter
img = cv2.Canny(cimg, t_lower, t_upper)
circles = cv2.HoughCircles(img,
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                param2 = 30, minRadius = 100, maxRadius = 200)

circles = np.uint16(np.around(circles))

a=0
b=0
r=0
r2=200

for i in circles[0,:]:
    # draw the outer circle
    #cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    #cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    if(i[2]>r and i[2]<r2):
        a=i[0]
        b=i[1]
        r=i[2]

#cv2.circle(cimg,(a,b),r,(0,255,0),2)
cv2.rectangle(cimg, (a - r, b - r), (a + r, b + r), (0, 128, 0), 4)
plt.subplot(121),plt.imshow(rgb_img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(cimg)
plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
plt.show()