# Author: Endri Dibra 
# Project: YOLOv12n Smoke and Fire Detection using Detection

# Importing the required libraries 
import os 
from ultralytics import YOLO


# Ensuring the correct path
os.chdir("C:/Users/User/Documents/AI_Robotics Projects/Smoke_Fire_Sensor_Fusion/Detection/YOLOv12n")

# Loading base model to train it on the datasets
# and be able to recognise smoke and fire cases
model = YOLO("yolo12n.pt")

# Starting training process
model.train(data="data.yaml", epochs=20, imgsz=640, seed=42)