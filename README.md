
| **Title**      |Face Recognition |
| ---------- |-------------------|
| **Team**       |Dang Quoc Vu - Vu. qvdang.96@gmail.com |
| **Predicting** |Build a model for Vietnamese football players face recognition. The data is collected from google and captured from videos. Need to build a model that predicts people in the large image.|
| **Data**       | capture images from videos (https://www.youtube.com), images from https://www.google.com/search|
| **Features**   |<ol> <li>image: continuous</li> <li>label: discrete</li> </ol>|
| **Models**     |<ol> <li>MMOD is used for human face detection.<br/>Link: http://dlib.net/files/mmod_human_face_detector.dat.bz2 </li> <li>Use VGG_Face_net as the model to get face emmbedding.<br/>It outputs 2622 embedding for each face image then we take this 2622 embeddings for later classification of image.<br/>Link:https://drive.google.com/uc?id=1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo
| **Future**     |impove the speed and accuracy in detecting and recognizing the human faces |
|**References**  |[1] https://arxiv.org/pdf/1502.00046.pdf <br/> [2] https://medium.com/analytics-vidhya/face-recognition-with-vgg-face-in-keras-96e6bc1951d5 <br/> [3] https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/ <br/> [4] https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/ 
| **Results**    |Score: > 0.96 |


## Directory structure
```
|cnn_model/
|  |-- mmod_face_detector.dat
|  |-- vgg_face_net_weight.h5
|dataset/
|  |--data_images/
|     |--video/
|        |--(video)
|     |--video_frames/
|        |--(frames of video)
|  |--test_image/ 
|     |--(images)
|  |--train_image/ 
|     |--Football-player[1]/
|        |--(images)
|     |--Football-player[2]/
|        |--(images)
|     |--Football-player[3]/
|        |--(images)
|     |--Football-player[4]/
|        |--(images)
|other_file/
|  |--(processed data files)
|src/
|  |--(files .py)
|test_result/
|  |--(images)
|trained_model/
|  |--(model file .h5)
```
## Command Line
```
```
Training:
```
python3 face_train.py --p <bool>
- True  : Process data before training 
- False : Use current processed data to train
```
```
Testing:
```
python3 face_test.py --image <image_name>
- Paste the test-image into /dataset/test_image/
- The result image is saved in test_result folder with the same name
``` 
```
Result:
```
![alt-text-1](https://github.com/qvdang-dev/Facial-Recognition/blob/develop/test_result/acc.png?raw=true "Accuracy chart") ![alt-text-2](https://github.com/qvdang-dev/Facial-Recognition/blob/develop/test_result/loss.png?raw=true "Loss chart")

The picture below shows the trained model is able to classify 11 football players in Vietnam national team successfully. 
![alt text](https://github.com/qvdang-dev/Facial-Recognition/blob/develop/test_result/U23VN.jpg?raw=true)
```

