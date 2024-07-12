<p align="center">
  <img src="readme_images/wired-gradient-69-eye.gif" width="100">
</p>

<h1 align="center">EduVision</h1>

<p align="center">
  <img src="https://img.shields.io/badge/made_by-kamawwanai-blueviolet?style=flat-square">
  <img src="https://img.shields.io/badge/vcpkg-used-blue?style=flat-square">
  <img src="https://img.shields.io/badge/opencv-v_4.8.0-green?style=flat-square">
  <img src="https://img.shields.io/badge/zlib-v_1.3.1-green?style=flat-square">
  <img src="https://img.shields.io/badge/sqlite3-v_3.46.0-lightgrey?style=flat-square">
  <img src="https://img.shields.io/badge/sqliteorm-v_1.8.2-lightgrey?style=flat-square">
  <img src="https://img.shields.io/badge/imgui-v_1.90.7-blue?style=flat-square">
</p>

<p align="center">
  <b>C++ application designed for automated student attendance tracking</b>
  <br><br>
  Developed as a part of my coursework, this project aims to simplify the process of recording and managing attendance data, making it efficient and error-free.
  <br>
  <div align="center">
  <img src="readme_images/Снимок экрана 2024-06-19 213935.png" width="500">
  </div>
</p>

<h2 align="center">Model Usage</h2>

<h3>Shape Predictor 68 Face Landmarks</h3>
The <a href="https://github.com/davisking/dlib-models/blob/41b158a24d569c8f12151a407fd1cee99fcf3d8b/shape_predictor_68_face_landmarks.dat.bz2">shape_predictor_68_face_landmarks.dat</a> model is widely used for detecting facial landmarks. It uses a method of cascaded regressors trained on a large dataset of annotated faces to identify 68 key points on the face.
<br><br>
Initially, the model detects faces in an image using the Histogram of Oriented Gradients (HOG) algorithm. HOG extracts brightness gradients and their directions, creating a feature vector for classifying regions as face or non-face. The image is normalized for brightness and contrast, gradients are computed for each pixel, and histograms of gradients are built for small cells. These histograms are then normalized and combined into a feature vector for classification.
<br><br>
After detecting faces, the model places an initial average face shape inside a rectangle around the detected face to set initial positions of the landmarks. The model iteratively adjusts these positions using texture features around current landmark positions. These features include brightness gradients and other image characteristics, aiding in precise landmark detection. The regressors, trained on a large dataset, minimize errors between current and true landmark positions, refining them iteratively until stable.
<br><br>
<h3>Dlib Face Recognition ResNet Model</h3>
The <a href="https://github.com/davisking/dlib-models/blob/41b158a24d569c8f12151a407fd1cee99fcf3d8b/dlib_face_recognition_resnet_model_v1.dat.bz2">dlib_face_recognition_resnet_model_v1.dat</a> uses a ResNet (Residual Network) architecture to convert face images into compact, informative vector representations (descriptors) for identification and comparison.
<br><br>
ResNet architecture includes residual blocks with convolutional layers, batch normalization, and ReLU activation. "Skip connections" in residual blocks help prevent the vanishing gradient problem, enhancing deep network training. Convolutional layers extract features at various abstraction levels, batch normalization stabilizes training, and ReLU introduces non-linearity for learning complex functions.
<br><br>
Pre-processing involves normalizing the face image to 150x150 pixels, adjusting brightness, and contrast. The ResNet converts this image into a 128-dimensional vector (descriptor) encoding key facial features. For face identification, the descriptor of a new image is compared to those in a database using Euclidean distance. If the distance is below a certain threshold, the faces are considered to match, indicating they belong to the same person.
<br>
<div align="center">
<img src="readme_images/accuracy.svg" width="800">
</div>
<br>
The model was tested on <a href="https://www.kaggle.com/datasets/bhaveshmittal/celebrity-face-recognition-dataset">a dataset from kaggle</a>
