# Devara_ML_Violence-detection
INTRODUCTION:
This project implements a violence detection system using deep learning techniques. It leverages Convolutional Neural Networks (CNN) for classification and YOLO (You Only Look Once) for object detection in video frames. The system can classify scenes as "Violent" or "Non-Violent" and is designed for use cases such as surveillance, content moderation, and safety monitoring.

OBJECTIVES:
Detect and classify frames or scenes in videos as "Violent" or "Non-Violent."
Incorporate object detection to enhance the system's understanding of scenes.
Build a real-time or batch processing system for efficient analysis.

DATASET:
A labeled dataset containing images or frames categorized as "Violent" and "Non-Violent."
Data preprocessing includes resizing images to 224×224, normalization, and data augmentation (flipping, rotation, etc.).

MODEL ARCHITECHTURE:
1. Violence Classification Model
Base Model: MobileNetV2, a pre-trained CNN, was used for feature extraction.
Custom Layers:
Global Average Pooling
Fully connected layers with dropout
Output layer with sigmoid activation for binary classification.
2. Object Detection
YOLOv8: A state-of-the-art object detection model was integrated to identify objects in video frames.

SYSTEM WORKFLOW:
Frame Extraction:
Video frames are extracted using OpenCV.
Each frame undergoes preprocessing (resizing, normalization).
Object Detection:
YOLO detects objects in each frame, providing bounding boxes and class labels.
Violence Classification:
Preprocessed frames are passed to the CNN model for violence detection.
Result Display:
Frames are annotated with bounding boxes, object labels, and violence classification results.
The output is saved as an annotated video or displayed in real-time

KEY FEATURES:
High Accuracy:
The system achieves competitive results using transfer learning and fine-tuning.
Real-Time Capability:
Integrated YOLO and CNN models enable real-time violence detection.
Scalability:
Can process videos offline or be deployed on edge devices and cloud platforms.

DEPLOYMENT:
Model Export:
The trained CNN model is saved as violence_model.h5.
YOLO weights are saved as yolov8s.pt.
Inference Pipeline:
Frame extraction, preprocessing, and classification are implemented in a Python script.
Output:
Annotated frames are displayed in real-time or saved as processed videos.

RESULTS:
The system effectively detects violence with high accuracy.
Metrics:
Accuracy: Achieved over 90% on the validation set.
Precision and Recall: Balanced for minimizing false positives and false negatives.
Visual results demonstrate clear detection and classification of violence in videos.

CONCLUSION:
This project successfully implements a robust violence detection system combining object detection and image classification. It is a scalable solution with potential applications in surveillance, public safety, and media monitoring.
