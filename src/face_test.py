import numpy as np
import os
import imutils
import dlib
import cv2
import pickle

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from utilities import get_folder_path, get_file_path, preprocess_image
from utilities import crop_rect_box_coodrs, get_face_detector_cnn, get_vgg_face_embedding

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_object_names(path):
    objects = []
    with open(path, 'rb') as f:
        objects = pickle.load(f)
    return (objects)

image_name = "vn3.jpg"
image_path = get_file_path('data_test', image_name)
image_result = get_file_path('result_test', image_name)
objects_path = get_file_path('other','object_names')
model_path = get_file_path('result_train','model_name')
face_detector = get_face_detector_cnn()
face_embedding = get_vgg_face_embedding(get_file_path('Pre_trained_CNN','vgg_model'))

image_org = cv2.imread(image_path)
image_gray = cv2.cvtColor(image_org, cv2.COLOR_RGB2GRAY)
coodrs = face_detector(image_gray, 1)
model = tf.keras.models.load_model(model_path)

objects = get_object_names(objects_path)
images, face_box = crop_rect_box_coodrs(coodrs,image_org,True)
print(face_box)
for i in range(images.shape[0]):
    box = face_box[i]
    face_encode = preprocess_image(images[i], face_embedding)
    face_embedded = K.eval(face_encode)
    y = model.predict(face_embedded)
    print(np.max(y))
    print(objects)
    if np.max(y) > 0.9:
        person = objects[np.argmax(y)]
    else:
        person = "unknow"
    cv2.waitKey(0)
    image_org = cv2.rectangle(image_org,(box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0,255,0), 1)
    cv2.putText(image_org, person, (box[0] + box[2], box[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,0,0), 1)

cv2.imwrite(image_result,image_org)
