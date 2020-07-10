import numpy as np
import pickle
import os
import imutils
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from face_embedding import get_vgg_face_embedding
import tensorflow.keras.backend as K

path = "../dataset/train_image/"
X = []
y = []
vgg_face_embedding = get_vgg_face_embedding()
objects = dict()
object_folders = os.listdir(path)

for i,object_index in object_folders:
    objects[i] = object_index
    images_path= path + str(object_index).zfill(2) + "/"
    images = os.listdir(images_path)
    for image_index in images:
        image = load_img(images_path + image_index, target_size=(224,224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        image_encode = vgg_face_embedding(image)
        X.append(np.squeeze(K.eval(image_encode)).tolist())
        y.append(object_index)
        print('done')
    

with open('train.pickle', 'wb') as f:
    pickle.dump([X, y], f)
