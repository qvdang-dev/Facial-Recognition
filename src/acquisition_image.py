import cv2
import numpy as np
import imutils
import dlib
import os
from face_detection_utilities import save_face_images

path = "../dataset/train_image/"
name = "qvdang"
path_data = ""

if not os.path.exists(path + name):
    os.makedirs(path + name)

path_data = path + name

cap = cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4,500)
faceDetector = dlib.cnn_face_detection_model_v1('../cnn_models/face_detector_00.dat')
while True:
    rat, image = cap.read()
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_box = faceDetector(image_gray, 1)
    index = save_face_images(face_box,image,True, path_data, name, index)
    index +=1
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    


