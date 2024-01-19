import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 0.01)

# prepare object points, like 90,0,0), (1,0,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d poins in image plane

images = glob.glob('*.jpeg')

for fname in images:
    img = cv2.imread(fname)
    img_original = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, update arrays
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw 
        img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        cv2.imwrite('img.jpg', img)
        # cv2.imshow('img', img)
        cv2.waitKey(0)



# calibrate camera 
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))       

# undistort
dst = cv2.undistort(img_original, mtx, dist, None, newcameramtx)
cv2.imwrite('undistorted.jpg', dst)
cv2.destroyAllWindows()

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: ", mean_error/len(objpoints))


#save the camera matrix and distortion matrix
np.savez("data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
npzfile = np.load('data.npz')
print(npzfile.files)



