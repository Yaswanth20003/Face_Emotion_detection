Note : model.h5 was not directly pushed to this repo due to more memory(150). 
        You can find model.h5 here "https://drive.google.com/file/d/1MPql4BPEmMBw9y0XJP66kk8SFxaTxG7j/view?usp=drive_link"


Face Emotion Detection System
Project Overview
This project is about detecting human emotions using their facial expressions with the help of Deep Learning (CNN model). The main purpose of this project is to help mentors track students' emotions during online classes like Google Meet. This helps mentors to know if students are happy, sad, or not paying attention.

Motivation
In SURE PROED, daily online classes are conducted. It is difficult for mentors to watch every student and understand their feelings. So, this project will help mentors to track students easily by detecting their emotions. In the future, this project can also be developed as a browser extension for online class platforms like Google Meet or Zoom.

Dataset Used
Dataset Name: MMAFEDB

File Location: /kaggle/input/mma-facial-expression/MMAFEDB

Total Emotions Detected: 7 Types
Angry
Disgust
Fear
Happy
Sad
Surprise
Neutral


Model Architecture
We used Convolutional Neural Network (CNN) for this project.

Layers Used:
Input Layer — (48x48x1) Image
Convolution Layers — To detect important features of the face
Batch Normalization — To speed up learning
Max Pooling — To reduce image size
Dropout — To avoid overfitting
Flatten Layer — To convert image to 1D array
Dense Layers — To learn patterns

Output Layer — To predict 7 emotions using Softmax

Results
Test Loss: 1.03
Test Accuracy: 65%

Web Application
Along with the Face Emotion Detection Model, developed a Web Application to make this system easy to use for everyone.

Tools & Technologies Used:
Frontend: HTML, CSS, JavaScript
Backend: Python (Flask Framework)
Model Integration: TensorFlow, Keras
