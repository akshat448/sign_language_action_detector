# Project Report: Sign language action detector with Streamlit app

## Introduction
This project focuses on the detection of Sign Language Actions using deep learning techniques to predict the actions performed by users based on live video input from a webcam or pre-recorded videos by using the Media Pipe framework created by Google.

## Data Collection
The data is collected using OpenCV(used for capturing video frames either from a webcam or pre-recorded video) and  Mediapipe Holistic is a framework developed by Google that 
provides pipelines that contain optimized face, hands, and pose components that allow for holistic tracking(in this project, only hand components are used).

By capturing the movements and positions of different hand landmarks from each frame of a video, this approach enables the creation of a dataset for training machine learning models in sign language recognition.

## Preprocessing:
### 1. Extracting Keypoints
An advantage of using media pipe over object detection is that the size of the data is reduced as media pipe only saves the location of the landmarks of our hands(x,y,z axis) instead of saving several images for the different hand signs.
used for training the machine learning model consists of a diverse set of sign language gestures. The dataset is preprocessed to extract relevant features from each frame of the videos and is stored in a numpy array, if any landmark is not present in a frame, 
then a numpy array of zeros is created in its place.


<p align="center">
  <img src="https://th.bing.com/th/id/OIP.fMBLvkdLbg0MEfv7KbJZjQAAAA?rs=1&pid=ImgDetMain" />
</p>

### 2. Creating folders for data collection
After the extraction of landmarks, the data is divided into folders under their class names depending on the action performed such as hello, yes, no, etc. 
This project only works for the following signs till now:

hello, thanks, love, yes, no, mother, father

### 3. Splitting data for training and testing
The collected data is split into training and testing sets along with their labels

## Model
### 1. Model architecture
A neural network with 3 Long Short-Term Memory (LSTM) is used as they are a type of recurrent neural network (RNN) known for handling sequential data effectively, along with 3 fully connected dense layers. The model is compiled using the Adam optimizer

### 2. Model Evaluation
For evaluation of the model, multilabel confusion matrices and the accuracy score of scikit-learn are used, in which the former is used to create confusion matrics for each class in dataset 
and the latter is used for calculating the accuracy of the model's predictions

### 3. Evaluating the model in real-time
Again, OpenCV along with media pipe is used to evaluate the model in real-time using a webcam and displaying the prediction on the camera feed window created by OpenCV.

## Streamlit Web Application
The user interface is built using Streamlit, a Python library for creating web applications with minimal code. 
The application allows users to select between live webcam input or uploading a video file. The predicted sign language action is displayed in real-time along with the width and frame rate(Only for webcam input) of the input video.

![image](https://github.com/akshat448/sign_language_action_detector/assets/129832161/f0d0fc49-181b-4764-a3eb-09b057d3dc85)

## Conclusion
The model achieves moderate accuracy in predicting sign language actions in real-time, it was a fun project that I may come back to, and I do have some plans for how I can make this better.
Further work can be done on the model to get better accuracy and for a larger number of classes. User testing has to be conducted to check the app's usability and user satisfaction, after which it can be deployed onto a cloud platform.

## Refrences
https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md
https://docs.streamlit.io/

