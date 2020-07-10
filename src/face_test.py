import numpy as np
import os
import imutils
import dlib
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from face_embedding import get_vgg_face_embedding
import tensorflow.keras.backend as K

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from face_detection_utilities import crop_rect_box_coodrs


path = "../cnn_models/face_detector_00.dat"
path_img_test = "../dataset/test_image/others/qvu.jpg"
path_folder = "../dataset/test_image/"


face_detector = dlib.cnn_face_detection_model_v1(path)
vgg_face = get_vgg_face_embedding()

image_org = cv2.imread(path_img_test)
image_gray = cv2.cvtColor(image_org, cv2.COLOR_RGB2GRAY)
coodrs = face_detector(image_gray)
model = tf.keras.models.load_model("../trained_model/model_00.h5")

object_folders = ["Bui-Tien-Dung", "Quoc-Vu", "Bang-Kieu", "Duy-Manh", "Nguyen-Thien-Nhan"]


images, face_box = crop_rect_box_coodrs(coodrs,image_org,True)
print(face_box)
for i in range(images.shape[0]):
    box = face_box[i]
    face_image = images[i]
    face_image = img_to_array(face_image)
    # print(face_image.shape)
    face_image = np.expand_dims(face_image, axis=0)
    # print(face_image.shape)
    face_image = preprocess_input(face_image)
    # print(face_image.shape)
    face_encode = vgg_face(face_image)
    face_embedded = K.eval(face_encode)
    y = model.predict(face_embedded)
    print(np.max(y))

    if np.max(y) > 0.6:
        person = object_folders[np.argmax(y)]
    else:
        person = "unknow"
    
    image_org = cv2.rectangle(image_org,(box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0,255,0), 2)
    cv2.putText(image_org, person, (box[0] + box[2], box[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

cv2.imwrite("../test_result/04.png" ,image_org)
