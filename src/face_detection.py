from face_detection_utilities import face_detecting_execute
import argparse

# construct the command parser to get arguments from the terminal
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmarks predictor")
    ap.add_argument("-i", "--image", required=True, help= "path of input image")
    ap.add_argument("--cnn", type = bool, help="pre-trained model for face detector: True : using CNN, False: using Default (HOG base)")
    ap.set_defaults(cnn=True)
    args = vars(ap.parse_args())
    return args

if __name__ == "__main__":
    args = get_args()
    face_detecting_execute(args)