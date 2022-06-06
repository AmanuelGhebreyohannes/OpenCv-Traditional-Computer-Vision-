import cv2
import numpy as np

video_src = 'pothole.mp4'

cap = cv2.VideoCapture(video_src)


while (cap.isOpened()):
    ret, img = cap.read()
    output = img.copy()
	

    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                param2 = 30, minRadius = 1, maxRadius = 40)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        #max_index = detected_circles[:,:,1]

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            #cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            cv2.rectangle(output, (a - r, b - r), (a + r, a + r), (0, 128, 255), -1)
        cv2.imshow("output", np.hstack([img, output]))
        #cv2.imshow("Detected Circle", img)
        cv2.waitKey(0)
