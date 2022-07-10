import csv
import os

import cv2
import torch.nn.modules.activation
import cv2 as cv
import mediapipe as mp
import numpy as np


def is_start(_image):
    action = False
    mp_holistic = mp.solutions.holistic

    # image = cv.imread(input_image)
    with mp_holistic.Holistic(model_complexity=0, min_tracking_confidence=0.1) as holistic:
        results = holistic.process(cv.cvtColor(_image, cv.COLOR_BGR2RGB))

        if results.pose_landmarks:

            noseY = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y

            left_wristY = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y - noseY
            right_wristY = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y - noseY

            left_shoulderY = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y - noseY
            right_shoulderY = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y - noseY

            left_crotchY = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y - noseY
            right_crotchY = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y - noseY

        else:

            left_wristY = 0
            right_wristY = 0

            left_shoulderY = 0
            right_shoulderY = 0

            left_crotchY = 0
            right_crotchY = 0

        shoulderY = (left_shoulderY + right_shoulderY) / 2
        crotchY = (left_crotchY + right_crotchY) / 2
        threashouldY = crotchY * 1.005

        if left_wristY < threashouldY or right_wristY < threashouldY:
            action = True

    return action


index = -1
find = False
cap = cv2.VideoCapture('D:/Software/PyCharm/AI/CSLdetection/video/007.avi')
while cap.isOpened():
    success, image = cap.read()
    index += 1
    if success:
        start_flag = is_start(image)
        print(start_flag)
        # if not find:
        #     start_flag = is_start(image)
        #     print(start_flag)
        #     if start_flag:
        #         find = True
        #         index = -1

        # else:
        #     print('666')

    # print(index)
