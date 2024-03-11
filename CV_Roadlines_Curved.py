import cv2
import numpy as np
import os
import pickle

# Image Paths
turns_folder = "C:\Paradigm\Paradigm_CV_2024\Computer-Vision-2024\Driving Pictures\Turns"
straights_folder = "C:\Paradigm\Paradigm_CV_2024\Computer-Vision-2024\Driving Pictures\Straights"
turns = []
straights = []

for image in os.listdir(turns_folder):
        turns.append(turns_folder + "/" + image)

for image in os.listdir(straights_folder):
        straights.append(straights_folder + "/" + image)
        

image = cv2.imread(straights[4])
calibration = cv2.imread("C:\Paradigm\Paradigm_CV_2024\Computer-Vision-2024\Calibration\Chessboard Calibration.jpg")

def showimage(image_title, img):
      image_resized = cv2.resize(img, (600,400))
      cv2.imshow(image_title, image_resized)

def calibrate(img):
    #This function uses a distorted chessboard to calibrate the camera matrix(mtx) and the distortion coefficients (dist)

    #Prepares Object Points
    obj_pts = np.zeros((5*9,3), np.float32)
    obj_pts[:,:2] = np.mgrid[0:9, 0:5].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.    objpoints = []
    imgpoints = []
    objpoints = []

    img_size = (img.shape[1], img.shape[0])

    #Convert Image to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,5), None)
    if ret == True:
        objpoints.append(obj_pts)
        imgpoints.append(corners)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)    
    
    undistorted_image = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted_image

showimage("original image", calibration)

undistorted = calibrate(calibration)
showimage("Undistorted", undistorted)

cv2.waitKey(0)
cv2.destroyAllWindows()