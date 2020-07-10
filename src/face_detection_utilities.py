import numpy as np
import cv2
import imutils
import dlib

NUM_OF_LANDMARKS = 68
FACE_SIZE = (224, 224)
index = 0

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
    

def face_detecting_execute(args):
    if args['cnn'] == False:
        # Initalize to face detector (HOG base)
        faceDetector = dlib.get_frontal_face_detector()
    else:
        # Initalize to CNN face detector
        faceDetector = dlib.cnn_face_detection_model_v1('../cnn_models/face_detector_00.dat')

    # Get the shape predictor
    shape_predictor = dlib.shape_predictor(args['shape_predictor'])

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