import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Load images
image1 = cv2.imread('left.png')
image2 = cv2.imread('right.png')

# Feature detection and matching (using ORB, for example)
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Extract matching points
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

# Homogeneous coordinates
points1 = np.column_stack((points1, np.ones(len(points1))))
points2 = np.column_stack((points2, np.ones(len(points2))))

# Essential matrix estimation
essential_matrix, _ = cv2.findEssentialMat(points1[:, :2], points2[:, :2], method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Recover pose (rotation and translation) from the essential matrix
_, R, t, _ = cv2.recoverPose(essential_matrix, points1[:, :2], points2[:, :2])

# Triangulate 3D points using the projection matrices
projection_matrix1 = np.hstack((np.eye(3), np.zeros((3, 1))))
projection_matrix2 = np.hstack((R, t.reshape(3, 1)))

# TriangulatePoints expects 2D points, so reshape points1 and points2
points1_2D = points1[:, :2].reshape(-1, 1, 2)
points2_2D = points2[:, :2].reshape(-1, 1, 2)

# Triangulate 3D points
points4D = cv2.triangulatePoints(projection_matrix1, projection_matrix2, points1_2D, points2_2D)
points3D = points4D[:3] / points4D[3]

# Plot 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points3D[0], points3D[1], points3D[2], c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()