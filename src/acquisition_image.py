import cv2
import numpy as np
import imutils
import dlib
import os
from utilities import save_face_images, get_folder_path, get_face_detector_cnn

object_ = {
    "name"  : "Bui-Tien-Dung",
    "video" : "Bui-Tien-Dung.mp4"
}

path = get_folder_path('video_frames') + "/" + object_['name']
video = get_folder_path('video') + "/" + object_['video']

if not os.path.exists(path):
    os.makedirs(path)

def get_video_frames(video, path):
    vidcap = cv2.VideoCapture(video)
    count = 0
    index = 0
    s = True
    while s:
        s, image = vidcap.read()
        if count == 50:
            image_path = path + "/" + str(index).zfill(5) + ".jpg"
            cv2.imwrite(image_path, image)
            count = 0
            index += 1

        cv2.imshow('frame', image)
        if cv2.waitKey(33) == ord('a'):
            break
        count += 1

def get_face_images(video_frames_path, face_image_path):
    save_path = face_image_path + "/" + object_['name'] 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    faceDetector = get_face_detector_cnn()
    images = os.listdir(video_frames_path)
    index = 0
    print(video_frames_path)
    for i,image_index in enumerate(images):
        image_path = video_frames_path + "/" + image_index
        print(image_path)
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        face_box = faceDetector(image_gray, 1)
        index = save_face_images(face_box,image,True, save_path, index)
        index +=1
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break

if __name__ == "__main__":
    
    get_video_frames(video, path)
    get_face_images(path,get_folder_path('data_images') )
    pass
        


