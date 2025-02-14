
import numpy as np
import cv2
import glob
import os

# Define checkerboard dimensions
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D points in real-world space
objpoints = []
# 2D points in image plane
imgpoints = []

# Prepare grid of points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Process images
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if ret:
        print(f"Checkerboard detected in {fname}")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    else:
        print(f"Checkerboard not detected in {fname}")

    cv2.imshow('img', img)
    cv2.waitKey(500)  # Display each image for 500ms

cv2.destroyAllWindows()

# Check if points were collected
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("No valid checkerboard images found. Exiting.")
    exit()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save results to a file
output_file = "results.npz"
np.savez(output_file, ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

print(f"Calibration results saved to {output_file}")
