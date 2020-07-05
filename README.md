Facial-Recognizance

command line:
$ python3 face_detection.py -p ../shape_predictor/68_face_landmarks.dat -i ../dataset/test_image/test-01.jpg --cnn True
usage: face_detection.py [-h] -p SHAPE_PREDICTOR -i IMAGE [--cnn CNN]

optional arguments:
  -h, --help            show this help message and exit
  -p SHAPE_PREDICTOR, --shape-predictor SHAPE_PREDICTOR
                        path to facial landmarks predictor
  -i IMAGE, --image IMAGE
                        path of input image
  --cnn CNN             pre-trained model for face detector: True : using CNN,
                        False: using Default (HOG base)


#1: Face detection
To get the better result we first need to detect only Faces in the image and then use the only faces image for recognition.
    - Apply the pre-trained model using HOG base (default function) or using CNN (cnn_models/face_detector_00.dat). to implement this work, we use "dlib".
    - Added number of functions for face detection - check in face_detection_utilities.py file.
    - Result: check images in test_result folder
