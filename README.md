Overview

LearnFlow AI is an intelligent classroom monitoring and student engagement analytics platform designed to improve online and digital learning experiences.

The system combines Computer Vision, Facial Landmark Tracking, Machine Learning, and Emotion Analysis to monitor student attention levels during lessons. Using a webcam, the platform continuously analyzes facial behavior, head movements, distraction patterns, and engagement indicators to generate real-time focus analytics and personalized learning reports.

The platform supports multiple user roles including students, teachers, and administrators, enabling educators to upload lessons, monitor learning sessions, review engagement reports, and identify students who may require additional support.

Unlike traditional learning management systems, LearnFlow AI actively measures learning behavior using AI rather than relying solely on quizzes or attendance records.

⭐ Key Features
👨‍🎓 Student Module
Student Registration & Login
Subject-wise Learning Dashboard
Lesson Access & Study Sessions
Real-Time Attention Monitoring
Webcam-Based Face Tracking
Learning Session Reports
Personalized Focus Recommendations
👨‍🏫 Teacher Module
Create Learning Materials
Upload Lesson Content
Subject Management
Monitor Student Learning Sessions
View Engagement Reports
Analyze Student Attention Statistics
🛠️ Admin Module
User Management
Student & Teacher Control
Subject Administration
System Monitoring Dashboard
User Activation/Deactivation
🤖 AI Monitoring Features
Real-Time Face Detection
Face Mesh Tracking using MediaPipe
Head Tilt Detection
Distraction Detection
Attention Classification
Learning Behavior Analytics
Confidence Score Generation
📊 Analytics & Reporting
Focus Score Analysis
Distraction Percentage
Concentration Metrics
Session Timeline Tracking
Personalized Study Recommendations
Historical Report Storage
😊 Emotion Intelligence Module

Additional DeepFace-based emotion analysis:

Engaged Student Detection
Frustration Detection
Confusion Detection
Boredom Detection
Low-Energy Detection

Emotion Mapping:

Detected Emotion	Learning State
Happy	Engaged
Neutral	Bored
Fear	Confused
Surprise	Confused
Angry	Frustrated
Disgust	Frustrated
Sad	Low Energy
💻 Technologies Used
Backend
Python
Flask
SQLite
Machine Learning
TensorFlow / Keras (.h5 Model)
Scikit-Learn
LabelEncoder
StandardScaler
Computer Vision
OpenCV
MediaPipe Face Mesh
Emotion Recognition
DeepFace
Data Processing
NumPy
JSON Processing
Authentication & Security
Werkzeug Password Hashing
Session Management
Frontend
HTML5
CSS3
JavaScript
Jinja2 Templates
Database
SQLite

Tables:

Users
Subjects
Lessons
Reports
🤖 Machine Learning Workflow
Step 1: Face Capture

Student webcam is activated.

Webcam
   ↓
OpenCV
   ↓
Video Frames
Step 2: Face Landmark Extraction

MediaPipe Face Mesh extracts facial landmarks.

Video Frame
      ↓
MediaPipe Face Mesh
      ↓
468 Face Landmarks
Step 3: Feature Engineering

Generated features:

Facial Landmark Coordinates
Head Tilt Angle
Face Orientation
Distraction Flag
Landmarks
      ↓
Feature Vector
      ↓
Head Tilt Analysis
Step 4: Data Preprocessing

Features are normalized using:

StandardScaler
Raw Features
      ↓
StandardScaler
      ↓
Normalized Features
Step 5: AI Prediction

Pre-trained Neural Network Model:

Normalized Features
       ↓
ANN Model (.h5)
       ↓
Attention State Prediction

Model Output:

Focused
Distracted
Engaged
Other Learning States

(Depends on training labels)

Step 6: Confidence Calculation
Prediction Probabilities
         ↓
Highest Probability
         ↓
Confidence Score
Step 7: Session Analytics

System stores:

Attention Events
Distraction Events
Focus Duration
Confidence Levels
Predictions
      ↓
Session Analytics
      ↓
Report Generation
Step 8: Final Report

Generated metrics:

Average Confidence
Concentration Score
Distraction Score
Timeline Analysis
Learning Recommendations
