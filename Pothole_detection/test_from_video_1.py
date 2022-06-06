import cv2
import numpy as np

video_src = 'pothole.mp4'

cap = cv2.VideoCapture(video_src)


while (cap.isOpened()):
    ret, img = cap.read()
    output = img.copy()
	
    
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    max_index = max(circles[:,:,1])
   
    # ensure at least some circles were found
    # if circles is not None:
    #     # convert the (x, y) coordinates and radius of the circles to integers
    #     circles = np.round(circles[0, :]).astype("int")
    #     # loop over the (x, y) coordinates and radius of the circles
    #     for (x, y, r) in circles:
    #         # draw the circle in the output image, then draw a rectangle
    #         # corresponding to the center of the circle
    #         cv2.circle(output, (x, y), r, (0, 255, 0), 4)
    #         cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    #     # show the output image
    #     cv2.imshow("output", np.hstack([img, output]))
    #     cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()