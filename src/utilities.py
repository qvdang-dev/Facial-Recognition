import os
import cv2
import dlib
import pickle
import imutils
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPool2D
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.model_selection import train_test_split


NUM_OF_LANDMARKS = 68
FACE_SIZE = (224, 224)
index = 0

Folders = {
    "data_train"        : "../dataset/train_image/",
    "data_test"         : "../dataset/test_image/",
    "Pre_trained_CNN"   : "../cnn_models/",
    "result_train"      : "../trained_model/",
    "result_test"       : "../test_result/",
    "other"             : "../other_files/"
}

Files = {
    "vgg_model"         : "vgg_face_weights.h5",
    "train_data"        : "train_data.pickle",
    "object_names"      : "objects.pickle",
    "model_name"        : "face_reg.h5"
}

def get_folder_path(folder_name):
    return  Folders[folder_name]

def get_file_path(folder_name, file_name):
    return Folders[folder_name] + Files[file_name]

def get_rect_box_coords(pos, value):
    if value:
        pos = pos.rect
    x = pos.left()
    y = pos.top()
    w = pos.right() - x
    h = pos.bottom() - y
    return (x, y, w, h)

def get_face_landmarks_coords(faceLandmarks):
    coords = np.zeros((NUM_OF_LANDMARKS, 2), dtype = int)
    for i in range(NUM_OF_LANDMARKS):
        coords[i] = (faceLandmarks(i).x, faceLandmarks(i).y)
    return coords

def show_rect_box_coodrs(coodrs, image_org, value):
    # show the output from the detected face coodirators

    for (i, face) in enumerate(coodrs):
        x, y, w, h = get_rect_box_coords(face, value)
        image_out = cv2.rectangle(image_org, (x, y), (x + w, y + h), (0,255,0), 1)
        # image_out = imutils.resize(image_out, width= 320, height=320)
        cv2.imshow('result',image_out)

def crop_rect_box_coodrs(coodrs, image_org, value):
    images_crop = []
    face_box = []
    for (i, face) in enumerate(coodrs):
        x, y, w, h = get_rect_box_coords(face, value)
        b = 50
        if x < b or y < b:
            b = 0
        image_crop = image_org[y-b:y+h+b, x-b:x+w+b]
        # image_crop = imutils.resize(image_crop, width=FACE_SIZE[0], height=FACE_SIZE[1])
        image_crop = cv2.resize(image_crop  , FACE_SIZE, interpolation = cv2.INTER_AREA)
        images_crop.append(image_crop)
        face_box.append((x-b, y-b, w+b, h+b))

    return np.array(images_crop), np.array(face_box)

def save_face_images(coodrs, image_org, value, path, name, index):
    # show the output from the detected face coodirators
    b = 0
    for (i, face) in enumerate(coodrs):
        x, y, w, h = get_rect_box_coords(face, value)
        if x > 0:
            image_out = image_org[y-b:y+h+b, x-b:x+w+b]
            image_out = imutils.resize(image_out, width=FACE_SIZE[0], height=FACE_SIZE[1])
            cv2.imwrite(path + "/"+ name + "_" + str(index) + ".png" ,image_out)
            index +=1
        # image_out = imutils.resize(image_out, width= 320, height=320)
    return index

def preprocess_image(image):
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    image_encode = vgg_face_embedding(image)
    return image_encode

def preprocess_data_from_path(path):
    X = [], y = []
    objects = dict()
    object_folders = os.listdir(path)
    vgg_face_embedding = get_vgg_face_embedding()
    for i,object_index in object_folders:
        objects[i] = object_index
        images_path= path + object_index + "/"
        images = os.listdir(images_path)
        for image_index in images:
            image = load_img(images_path + image_index, target_size=FACE_SIZE)
            image_encode = preprocess_image(image)
            X.append(np.squeeze(K.eval(image_encode)).tolist())
            y.append(i)
    with open(get_file_path('other','train_data'), 'wb') as f:
        pickle.dump([X, y, objects], f)   
    with open(get_file_path('other','object_names'), 'wb') as f:
        pickle.dump(objects, f)
    return

def face_detecting_execute(args):
    if args['cnn'] == False:
        # Initalize to face detector (HOG base)
        faceDetector = dlib.get_frontal_face_detector()
    else:
        # Initalize to CNN face detector
        faceDetector = dlib.cnn_face_detection_model_v1('../cnn_models/face_detector_00.dat')

    # Get the shape predictor
    # shape_predictor = dlib.shape_predictor(args['shape_predictor'])

    cap  = cv2.VideoCapture(0)
    cap.set(3, 480)
    cap.set(4, 320)
    while True:
        ret, image = cap.read()
        # Preprocess the input image: convert to gray img, resize
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # first parameter: type of image - RGB or Gray image
        # second parameter: number of image's layers
        face_box = faceDetector(image_gray, 1)
        show_rect_box_coodrs(face_box, image, args['cnn'])
        # press ESC to break the LOOP
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break


""" 
    Our pre-trained model file (".h") just includes model weigths 
    and nothing else so we need to reconstruct the whole architecture 
    of it (VGG-FACE network)
"""
def get_vgg_face_embedding():

    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape = (224, 224, 3)))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3,3), activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3,3), activation='relu' ))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3,3), activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3,3), activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3,3), activation='relu' ))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3,3), activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3,3), activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3,3), activation='relu' ))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3,3), activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3,3), activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3,3), activation='relu' ))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(4096, (1,1), activation='relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(2622, (1,1), activation='relu'))
    model.add(Flatten())
    model.add(Activation('softmax')) 

    model.load_weights(path)
    # skip the final layer(Activatio = softmax)
    # use Flatten layer for the input of the FC (constucted in training)
    vgg_face = Model(inputs =model.layers[0].input, outputs = model.layers[-1].output)
    return vgg_face

