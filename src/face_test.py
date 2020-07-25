import os
import argparse
import imutils
import dlib
import cv2
import pickle
import numpy as np
from PIL import Image

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
fontPath = "/usr/share/fonts/truetype/freefront"
def get_argument():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=str, help="Get test-image path")
    args = ap.parse_args()
    return args
def get_object_names(path):
    objects = []
    with open(path, 'rb') as f:
        objects = pickle.load(f)
    return (objects)

def face_recognize(image):
    image_path = get_file_path('data_test', image)
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
    image_org = Image.fromarray(cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB))
    for i in range(images.shape[0]):
        box = face_box[i]
        face_encode = preprocess_image(images[i], face_embedding)
        face_embedded = K.eval(face_encode)
        y = model.predict(face_embedded)
        if np.max(y) > 0.7:
            person = objects[np.argmax(y)]
        else:
            person = "*"
        print(np.max(y), person)

        # cv2.waitKey(0)
        # image_org = cv2.rectangle(image_org,(box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0,255,0), 1)

        fort = ImageFont.truetype(fontPath + "/" + "FreeMonoBold.ttf", size=60)
        pos_name = (box[0] + box[2], box[1] - 5)
        pos_box = (box[0] + box[2] -1, box[1] - 5)
        pos_rect = (box[0], box[1]), (box[0] + box[2], box[1] + box[3])
        
        if person != "*":
            tw, th = fort.getsize(person)
            canvas = Image.new('RGB', (int(tw/5) - 10, int(th/5) + 1), "orange")
            image_org.paste(canvas, pos_box)
        
        draw = ImageDraw.Draw(image_org)
        if person != "*":
            draw.text(pos_name,person, 'blue', fort=fort)
        draw.rectangle(pos_rect, outline='green')

    image_org.save(image_result)

if __name__ == "__main__":
    args = get_argument()
    image_name = args.image
    face_recognize(image_name)
    pass