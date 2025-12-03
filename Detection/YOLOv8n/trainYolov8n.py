# Author: Endri Dibra 
# Project: YOLOv8n Smoke and Fire Detection using Detection

# Importing the required libraries 
import os 
from ultralytics import YOLO


# Ensuring the correct path
os.chdir("C:/Users/User/Documents/AI_Robotics Projects/Smoke_Fire_Sensor_Fusion/Detection/YOLOv8n")

# Loading base model to train it on the datasets
# and be able to recognise smoke and fire cases
model = YOLO("yolov8n.pt")

# Starting training process
model.train(data="data.yaml", epochs=20, imgsz=640, seed=42)