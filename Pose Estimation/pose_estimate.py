import cv2
import numpy as np
import glob

# load previously saved data
with np.load("data.npz") as X:
    mtx, dist, _ , _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

# draw axis
def draw(img, corners_ref, imgpts):
    corner_ref = tuple( int(element) for element in corners_ref[0].ravel())
    img = cv2.line(img, corner_ref, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner_ref, tuple(imgpts[1].ravel()), ( 0, 255, 0), 5)
    img = cv2.line(img, corner_ref, tuple(imgpts[2].ravel()), ( 0, 0, 255), 5)
    return img

# draw a cube
def draw_cube(img, corners, imgpts):
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i,j in zip(range(4), range(4,8)):
        img = cv2.line(img,tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0,0,255), 3)

    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 0.01)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0, 0, -3]]).reshape(-1,3)


for fname in glob.glob('img*.jpeg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    if(ret == True):
        corners2 = cv2.cornerSubPix(gray, corners,(11,11), (-1,-1), criteria)

        # Find the rotation and translation vectors
       
        flag, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        imgpts = np.int32(imgpts).reshape(-1, 2)
        corners2 = np.int32(corners2).reshape(-1,2)
        #draw axis
        img = draw(img, corners2, imgpts)
        cv2.imshow('img', img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite("imgaxis.jpg", img)

        # draw cube
        axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        imgpts = np.int32(imgpts).reshape(-1, 2)  # cast to int from float
        img = draw_cube(img, corners2, imgpts)
        cv2.imshow('img', img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite("imgaxis.jpg", img)

cv2.destroyAllWindows()