import cv2
import time
import numpy as np


# Optical Flow

lk_params = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 20, qualityLevel = 0.3, minDistance = 10, blockSize = 7)

trajectory_len = 20
detect_interval = 1
trajectories = []
frame_idx = 0

cap = cv2.VideoCapture(0)

while True:
    start = time.time()
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()
    

# Centroid Tracking





