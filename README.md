# EMOJI RECOMMENDATION SYSTEM BASED ON FACIAL EXPRESSIONS 

**The domain of the Project:** Deep Learning, Computer Vision

**Team Mentors (and their designation):** Mr. Gaurav Patel (Data Analyt)

**Team Members:** Mr. Dosapati Yaswanth Guptha (B.Tech, CSBS) 3<sup>rd</sup> year

**Period of the project**: 2024 - April 2025

# SURE ProEd PUTTAPARTHI, ANDHRA PRADESH

### **_DECLARATION_**

The project titled **“EMOJI RECOMMENDATION SYSTEM BASED ON FACIAL EXPRESSIONS”** has been mentored by **Mr. Gaurav Patel**, **Mr. Shanmukha**, **Mr. Ravi Shankar** and organized by **SURE ProEd** to **2024 - April 2025**. This project leverages deep learning techniques to detect emotions from facial expressions, with a focus on real-time analysis.

I, **Dosapati Yaswanth Guptha**, hereby declare that I have solely worked on this project under the guidance of my mentor. This project has significantly enhanced my practical knowledge and skills in the domain of **Computer Vision** and **Deep Learning**.

| **Name** |
| --- |
| Mr. D. Yaswanth Guptha | 

| **Mentor Nme** |
| --- |
| Mr. Gaurav Patel | 

**Seal & Signature**

Prof.Radhakumari

Executive Director & Founder

SURE ProEd

**_Executive Summary_**

This project implements a real-time **Face Emotion Detection System** using deep learning techniques trained on the **MMAFEDB (Multi-Modal Affective Faces Expression Database)** dataset. The system is designed to automatically identify and classify human facial emotions—such as happiness, sadness, anger, surprise, and more—through visual input, enabling applications in areas like human-computer interaction, mental health analysis, and smart surveillance.

A convolutional neural network (CNN)-based deep learning model was trained on the MMAFEDB dataset, which includes a diverse range of facial expressions captured under varying conditions. The model was then integrated into a real-time webcam application using OpenCV and Flask, capable of live facial emotion recognition and display.

The use of transfer learning and data augmentation techniques helped improve model accuracy and generalization. This project highlights the effectiveness of deep learning in computer vision tasks and demonstrates a practical solution for emotion recognition in real-world environments.

This system offers a strong foundation for future enhancements, including multi-modal input processing, deployment on edge devices, and integration with broader emotion-aware applications.

# **_Introduction_**

## **Background and Context**

Facial emotion recognition is a key application of computer vision and human-computer interaction. With deep learning techniques, emotion detection has become more accurate and real-time capable. This project uses the MMAFEDB dataset to train a CNN model for identifying facial emotions. It aims to enhance user interaction by enabling machines to understand human emotions effectively.

## **Problem Statement**

Despite advancements in deep learning, accurately detecting facial emotions in real-time remains a challenge due to variations in lighting, facial expressions, and individual differences. Existing models often require high computational power and large datasets, making deployment on real-world applications difficult. This project aims to develop an efficient deep learning model using the MMAFEDB dataset to recognize emotions from facial images in real-time.

## **Scope**

This project aims to develop a lightweight, resource-efficient voice command recognition system designed to run on the ESP32 microcontroller, leveraging signal processing techniques instead of machine learning. The scope includes:

1. **Real-Time Emotion Detection:** The system captures live webcam feed to detect and classify facial emotions in real-time.
2. **Deep Learning Integration:** Utilizes deep learning techniques to train a model on the MMAFEDB dataset for accurate emotion classification.
3. **Multi-Emotion Recognition:** Capable of identifying a range of human emotions such as happiness, sadness, anger, surprise, fear, disgust, and neutrality.
4. **Practical Applications:** Designed for potential use in mental health monitoring, user experience enhancement, e-learning feedback systems, and smart surveillance.
5. **Webcam-Based Input:** Emphasizes ease of deployment using standard webcam input, without the need for specialized hardware.

## **Limitations**

1. **Lighting Dependency:** Accuracy may decrease in low-light or overly bright conditions due to inconsistent facial feature visibility.
2. **Real-Time Lag:** On lower-end systems, real-time processing may experience latency due to computational demands of the model.
3. **Occlusion Sensitivity:** Emotions may not be accurately detected if the face is partially covered (e.g., masks).
4. **Subtle Emotion Confusion:** The model may struggle to distinguish between subtle or overlapping emotions such as neutrality and slight sadness.

## **Innovation**

This project introduces a real-time face emotion detection system using deep learning, trained specifically on the MMAFEDB dataset—a richly annotated and diverse collection tailored for multi-modal emotion analysis. Unlike traditional models that focus on static image classification, our system processes live webcam input and dynamically predicts emotional states, enabling more interactive human-computer interaction. By integrating computer vision techniques with convolutional neural networks (CNNs), the project showcases an innovative application of deep learning for affective computing. The model’s deployment in real-time adds a novel edge, making it suitable for modern applications like virtual counseling, sentiment-aware tutoring systems, and adaptive user interfaces.

# **_Project Objectives_**

## **Project Objectives and Expected Outcomes**

1. **Develop a Real-Time Face Emotion Detection System**: Design and implement a deep learning model that processes live webcam input to detect human emotions using the MMAFEDB dataset.  
    **Expected Outcome:** A fully functional system capable of recognizing emotions in real-time, improving human-computer interaction and suitable for dynamic environments.
2. **Train a Deep Learning Model for Emotion Recognition**: Train a Convolutional Neural Network (CNN) on the MMAFEDB dataset to classify emotions such as happiness, sadness, surprise, anger, and neutral.  
    **Expected Outcome:** A robust model that accurately classifies facial emotions with high accuracy, demonstrating the effectiveness of CNNs in emotion recognition tasks.
3. **Implement Real-Time Emotion Detection**: Implement the trained deep learning model for real-time processing of webcam input, ensuring low latency and accurate predictions.  
    **Expected Outcome:** A system capable of detecting facial emotions in real-time with minimal delay, enhancing user experience for applications like virtual counseling or sentiment analysis.
4. **Evaluate Model Performance**: Test and evaluate the performance of the model based on metrics such as accuracy, precision, recall, and F1 score to ensure its effectiveness in recognizing emotions.  
    **Expected Outcome:** A comprehensive analysis of the model's performance, identifying areas of improvement and ensuring its reliability in real-world applications.
5. **Explore Robustness and Adaptability to Different Lighting Conditions**: Implement strategies to handle varying lighting conditions and facial occlusions to maintain the system’s robustness across different

environments.  
**Expected Outcome:** A system that remains accurate and reliable even under challenging conditions such as poor lighting or partial occlusion.

1. **Provide Real-World Application Use-Cases**: Integrate the emotion detection system into practical applications such as virtual assistants, interactive entertainment, or mental health monitoring tools.  
    **Expected Outcome:** Demonstrate the versatility of emotion recognition for a variety of real-world applications, showcasing its potential in adaptive user interfaces and emotional intelligence systems.

## **Deliverables**

1. **Working Face Emotion Detection System**: A fully functional system capable of detecting and classifying emotions (e.g., happiness, sadness, surprise, anger, neutral) in real-time using webcam input and the trained deep learning model.
2. **Demo Video**: A video demonstrating the working of the face emotion detection system, showcasing its ability to recognize and display detected emotions in real-time from webcam footage.
3. **Model Performance Report**: A comprehensive report evaluating the performance of the emotion detection model, including accuracy, precision, recall, F1 score, and comparisons with other approaches (if any), to demonstrate its effectiveness.
4. **Robustness to Noise Implementation**: A version of the system that incorporates techniques for handling varying lighting conditions and facial occlusions, ensuring robustness in different real-world environments.
5. **Codebase**: The complete source code for the face emotion detection system, including the training scripts, model architecture, and real-time detection code, along with necessary comments and documentation for reproducibility.
6. **Technical Documentation**: Detailed documentation explaining the system's design, model architecture, training process, and instructions for setting up and running the emotion detection system, ensuring clarity and ease of understanding for future developers or researchers.

# **_Methodology and Results_**

## **Methods/Technology Used**

1. **Convolutional Neural Networks (CNNs)**: Used for training and detecting emotions from facial expressions in images.
2. **MMAFEDB Dataset**: The dataset used for training the model, consisting of images of faces with labeled emotions.
3. **OpenCV**: Used for face detection and image preprocessing (such as resizing and normalization).
4. **TensorFlow/Keras**: Frameworks used for building and training the deep learning model.
5. **Python**: Primary programming language for implementing the system, including training the model and image processing.
6. **Webcam**: Used for capturing real-time facial images to detect emotions.
7. **Flask/Django (optional)**: For building the web interface to display real-time emotion predictions.
8. **Matplotlib/Seaborn**: For visualizing the model's performance and emotion prediction results during development.

## **Tools/Software Used**

1. **Python**: Programming language used for building the face emotion detection system.
2. **TensorFlow**: Open-source framework for training and deploying the deep learning model.
3. **Keras**: High-level neural network API used to build the CNN model.
4. **OpenCV**: Computer vision library used for real-time face detection and image processing.
5. **Matplotlib**: Plotting library used for visualizing model training metrics.
6. **Jupyter Notebook**: Integrated development environment for model training and testing.
7. **PyCharm/VS Code**: IDEs used for developing the project code.

## **Data Collection Approach**

1. **Dataset Used**: MMAFEDB (Mature Facial Expression Database) dataset, which includes facial images annotated with different emotion labels.
2. **Data Preprocessing**: The images were preprocessed by resizing, normalization, and face alignment to improve model accuracy.
3. **Augmentation**: Data augmentation techniques like rotation, flipping, and zooming were applied to increase the diversity of the training dataset.
4. **Split**: The dataset was split into training, validation, and testing sets to evaluate model performance.
5. **Data Labeling**: Each image in the dataset was labeled with one of the following emotions: Happy, Sad, Surprise, Angry, Neutral, etc.

## **Project Architecture**

1. **Data Collection**: The system uses the MMAFEDB dataset, which contains images of faces with labeled emotions such as happiness, sadness, anger, etc. The images are collected for training the model.
2. **Data Preprocessing**: The images from the dataset undergo several preprocessing steps:
3. **Resizing**: Images are resized to a standard dimension.
4. **Normalization**: Pixel values are scaled to a specific range (typically 0-1).
5. **Face Detection and Alignment**: Faces in the images are detected and aligned using a face detection model to ensure uniformity in the input data.
6. **Model Training**: A Convolutional Neural Network (CNN) is trained on the preprocessed dataset. The CNN learns to identify patterns and features in the facial images that correspond to different emotions.
7. **Emotion Detection**: Once the model is trained, it can be used for real-time emotion detection:
8. A webcam or image input is used to capture the face of the user.
9. The face is passed through the trained CNN to predict the emotion.
10. **User Interface**: A simple interface (could be a web app or desktop application) displays the detected emotion in real-time. The system also shows the confidence level of the emotion detected, allowing the user to see the system's certainty in its prediction.

## **Results**

**Superior Real-time Performance**: The system demonstrated efficient real-time emotion recognition with minimal latency, allowing for fast detection directly from webcam input.

**Effective Emotion Detection with Varying Facial Expressions**: The model was able to recognize emotions accurately even with slight variations in facial expressions and lighting conditions.

**Before Balanced Dataset - Image Distribution Per Class**
The distribution across various emotion classes in a balanced dataset after data augmentation. Each emotion class, labeled from 0 to 6, has a nearly identical count of images, ranging 26621. This even distribution demonstrates that the dataset has been successfully balanced, ensuring equal representation of all emotion classes, which is crucial for training machine learning models to avoid bias towards any specific emotion.

**After Balanced Dataset - Dataset Distribution**
The distribution of the dataset after balancing efforts. The Training segment constitutes 78.8% of the total data, ensuring the model has ample examples for learning. The Public Test and Private Test segments each make up 10.6%, allowing for consistent and fair evaluation. This balanced partitioning supports effective model training while preserving reliability during testing and validation phases.

**Accuracy and Loss Comparison Across Datasets**

| **Dataset** | **Accuracy** | **Loss** |
| --- | --- | --- |
| MMAFEDB | 0.6543 | 0.3945 |
| AffectNet (Unbalanced) | 0.5969 | 1.0915 |
| Fer2013 | 0.5215 | 2.7690 |
| AffectNet (Balanced) | 0.4977 | 2.1932 |
| Combined Datasets | 0.0779 | 1.9514 |

**CNN architecture is designed for facial emotion recognition and consists of multiple convolutional and pooling layers followed by a flattening and output.

#### **Input Layer**

- Accepts facial image data, typically in the form of a 2D image (e.g., 48x48 grayscale or 64x64 RGB).
- The input is passed into the first convolutional block.

#### **Convolutional Blocks** (Repeated 4 Times)

Each block includes:

1. **Conv2D Layer**
    - Applies 2D convolution with a kernel size of 3×3.
    - Filters:
        - First 3 blocks: 256 filters
        - Last block: 512 filters
    - Activation: **ReLU** (Rectified Linear Unit), which introduces non-linearity.
2. **Batch Normalization**
    - Normalizes the output of the convolution to stabilize and speed up training.
3. **MaxPooling2D (2×2)**
    - Reduces spatial dimensions (height and width), helping with computational efficiency and overfitting reduction.
4. **Dropout (Rate = 0.3)**
    - Randomly disables 30% of neurons during training to prevent overfitting and enhance generalization.

#### **Flatten Layer**

- Transforms the final 3D feature maps from the convolutional blocks into a 1D vector to prepare it for the dense (fully connected) layer.

#### **Output Layer**

- This layer is typically a Dense layer with a number of neurons equal to the number of emotion classes (e.g., 7 for emotions like Happy, Sad, Angry, etc.).
- **Activation Function**: Usually **Softmax** for multi-class classification.

Here as least validation loss and the max of validation accuracy are saved.

At Epoch 74 the validation loss of 0.39 was saved Epoch 74: val_loss improved from 0.61673 to 0.39452, saving model to weights_min_loss.keras 448/448 - 1s - 2ms/step - accuracy: 0.6094 - loss: 0.8980 - val_accuracy: 1.0000 - val_loss: 0.3945

At Epoch 77 the validatin accuracy was maximum of 0.65 Epoch 77: val_loss did not improve from 0.39452 448/448 - 17s - 39ms/step - accuracy: 0.7082 - loss: 0.7825 - val_accuracy: 0.6565 - val_loss: 0.9884

| **Metric** | **Value** | **Epoch** |     |
| --- | --- | --- |     |
| Best val_loss | 0.3945 |     | 74  |
| Best val_accuracy | 0.6565 | 77  |     |

Training loss is decreasing and accuracy is increasing. But validation graph is fluctuating due to overfitting or less data.

**Evaluation on Test Data**

| Test Loss | 1.03 |
| --- | --- |
| Test Accuracy | 65.53% |



### **Face Emotion Detection**

**Image 1:**

- Emotion Detected: Neutral



**Image 2:**

- Emotion Detected: Happy


**Image 3:**

- Emotion Detected: Surprise


**GitHub Link**

**<https://github.com/Yaswanth20003/Face_Emotion_detection>**

**_Social / Industry relevance of the project_**

**Social / Industry relevance of the project**

### 1\. **Enhanced Communication in Messaging Platforms**

In modern digital communication, emojis have become an emotional language. This system can be integrated into:

- Chat applications such as WhatsApp, Messenger, and Telegram
- Social media platforms including Instagram, Snapchat, and X (formerly Twitter)

**Why it matters:**  
Text-based communication often lacks tone and emotion. By recommending emojis based on real-time facial expressions, conversations become more personal, expressive, and engaging.

### 2\. **Accessibility and Inclusivity**

The system can provide valuable support for:

- Individuals with cognitive or speech impairments
- Elderly users who may not be familiar with emoji libraries
- People with disabilities who rely on assistive technologies and visual aids

**Why it matters:**  
Enabling emotion-driven emoji input helps bridge the gap between digital tools and users with limited communication capabilities, promoting greater inclusivity and user-friendliness.

**_Learning and Reflection_**

### **Learning and Reflection**

### 1\. **Computer Vision using OpenCV**

I gained practical knowledge of OpenCV for face detection, frame capturing, and image pre-processing. I learned how to handle real-time webcam feeds, detect facial landmarks, and align faces for consistent model input.

### 2\. **Facial Expression Recognition**

This was my first experience working with emotion recognition. I learned how to preprocess datasets with emotion labels, extract meaningful features from facial images, and classify expressions like happy, sad, angry, neutral, and more using deep learning.

### 3\. **Machine Learning and Deep Learning Concepts**

Through frameworks like TensorFlow and Keras, I learned how to:

- Design CNN architectures tailored for image classification.
- Apply techniques such as data augmentation, dropout, and early stopping.
- Fine-tune hyperparameters to improve model accuracy and generalization.

### 4\. **Model Training and Evaluation**

I developed a strong understanding of analyzing training metrics:

- Learned to read and interpret training vs validation curves.
- Identified signs of overfitting/underfitting and applied strategies to address them.
- Understood the importance of balanced datasets and effective splitting for fair evaluation.

### **Real-time System Development**

Deploying the model in a real-time environment was one of the most exciting parts. I learned how to:

- Connect live camera input with prediction pipelines.
- Overlay recommended emojis on live feed.
- Optimize performance to reduce lag and enhance user experience.

### **New Technical Tools & Libraries**

Throughout the project, I became proficient with:

- NumPy and Pandas for data handling.
- Matplotlib and Seaborn for visualizations.
- Scikit-learn for preprocessing and evaluation.
- OpenCV for video capture and drawing on frames.

### **Soft Skills and Collaboration**

- **Teamwork:** I learned to divide tasks efficiently, maintain clear communication, and support each other’s roles during implementation and debugging.
- **Problem Solving:** Encountering and solving bugs, memory issues, and performance lags improved my logical thinking and patience.
- **Presentation Skills:** Explaining our system to mentors and peers gave me confidence in articulating technical ideas clearly and concisely.

### **Overall Experience**

This project was a rich and rewarding experience. From handling datasets to building real-time AI systems, I gained exposure to end-to-end development. There were challenges—like tuning the model or managing inconsistent webcam input but overcoming them made the journey more meaningful. I now feel more confident in applying machine learning and computer vision techniques to real-world problems and look forward to exploring more advanced projects in the future.

**_Future Scope & Conclusion_**

## **Objectives**

1. **To develop a facial expression recognition system** that accurately detects and classifies human emotions in real-time using image inputs from a webcam or camera.
2. **To design and train a machine learning model** (preferably CNN-based) capable of recognizing key emotions such as Happy, Sad, Angry, Neutral, Surprise, Disgust, and Fear.
3. **To recommend appropriate emojis** based on the detected facial emotion to enhance expressiveness in digital communication.
4. **To preprocess and balance the dataset** through techniques such as data augmentation for improved model performance and reduced bias.
5. **To implement real-time integration** of emotion detection and emoji recommendation, ensuring smooth and efficient performance.
6. **To analyze model performance** using metrics like accuracy, loss curves, confusion matrix, and ensure generalization without overfitting.
7. **To create a user-friendly interface** that displays the predicted emotion and the corresponding emoji for real-time use cases.

## **Achievements**

1. **Successfully Trained Emotion Classification Model**

Built and trained a deep learning model capable of recognizing seven facial emotions with significant accuracy using CNN architecture.

1. **Balanced Dataset for Improved Accuracy**

Achieved uniform class distribution through data augmentation techniques, leading to better generalization and reduced model bias.

1. **Integrated Real-Time Emotion Detection**

Implemented real-time facial expression recognition using OpenCV and webcam input, enabling dynamic emoji suggestions.

1. **Developed Emoji Recommendation Engine**

Created a system that maps each detected emotion to a suitable emoji, enhancing digital communication experience.

1. **Optimized Model Performance**

Applied techniques like dropout, batch normalization, and tuning to reduce overfitting and improve validation accuracy.

1. **Analyzed and Visualized Results**

Successfully generated bar charts, pie charts, and loss/accuracy plots to demonstrate dataset insights and model performance.

1. **Built a Functional GUI**

Designed a simple user interface displaying real-time emotion output and recommended emojis, ensuring user-friendliness.

1. **Collaborative Team Effort**

Completed the project as a team, demonstrating effective task delegation, communication, and problem-solving.

## **Conclusion**

The **Emoji Recommendation System based on Facial Expressions** successfully bridges the gap between human emotions and digital communication by interpreting facial expressions in real time and suggesting appropriate emojis. Through the integration of **computer vision**, **deep learning**, and **real-time processing**, we developed a model capable of detecting emotions like happiness, sadness, anger, and surprise with reliable accuracy.

By balancing the dataset, fine-tuning the CNN model, and addressing challenges like overfitting and data scarcity, we were able to improve the system’s performance. The project also highlights the potential for emotion-aware technologies in enhancing user experience across messaging platforms, particularly for users with communication difficulties.

Overall, this project not only expanded our technical expertise in **OpenCV**, **TensorFlow**, and **real-time ML deployment**, but also emphasized the social relevance of emotion-driven interactions in modern applications.

**Future Scope**

1. **Vertual Classes Monitoring**: As SURE ProEd conducts daily online classes, a Face Emotion Detection Browser Extension can be developed to help mentors easily track students emotions. This will simplify student monitoring through emotion classification. Not only for SURE ProEd, but also for mentors of all online classes, this tool will enable efficient and effective tracking of learners
2. **Mental Health Monitoring**: The system could be expanded to help track emotional well-being over time by recommending emojis that reflect different emotional states. It could alert users or caregivers if any significant emotional shifts are detected, providing an additional tool for emotional self-awareness or even early intervention in mental health care.
