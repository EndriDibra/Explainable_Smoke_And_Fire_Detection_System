# Author: Endri Dibra 
# Project: Smoke and Fire Detection with YOLOv8n using Detection

# Importing the required libraries
import os
import cv2
import time
import csv
from ultralytics import YOLO


# Setting Input and Output folders
INPUT_FOLDER = "Input_Images"
OUTPUT_FOLDER = "Output_Images"

# Creating output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Creating CSV file for saving all results
csvFilePath = os.path.join(OUTPUT_FOLDER, "results.csv")

# Creating CSV and writing header
with open(csvFilePath, mode="w", newline="") as file:
 
    writer = csv.writer(file)
    writer.writerow(["Image Name", "FPS", "Fire Scores", "Smoke Scores"])


# Loading YOLOv8n best model
model = YOLO("runs/detect/train/weights/best.pt")  

# Getting list of image files
imageFiles = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

# Checking if the input folder contains images or it is empty
if len(imageFiles) == 0:
    
    print("No images found in Input_Images folder.")
    
    exit()

# Processing input images
for imageName in imageFiles:
   
    imgPath = os.path.join(INPUT_FOLDER, imageName)
   
    print(f"Processing: {imageName}")

    # Reading image
    image = cv2.imread(imgPath)
   
    # Checking if image can be ridden
    if image is None:
   
        print(f"Could not read {imageName}, skipping...")
   
        continue

    # Resizing original image to 640 x 640
    image = cv2.resize(image, (640, 640))

    # FPS calculation (for processing speed, not real-time)
    startTime = time.time()

    # Running YOLOv8n best model detection
    results = model.predict(

        source=image,
        conf=0.5,      
        imgsz=640,
        verbose=False
    )

    det = results[0]

    # storing conf scores for fire
    fireScores = []     

    # storing conf scores for smoke
    smokeScores = []    

    # Extracting all fire and smoke detection scores
    for box in det.boxes:
        
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Fire class
        if cls == 0:       
            
            fireScores.append(conf)

        # Smoke class
        elif cls == 2:     
            
            smokeScores.append(conf)


    # Drawing results on the frame
    annotatedImage = det.plot()

    fps = 1 / (time.time() - startTime)

    # Add FPS text to the frame
    cv2.putText(annotatedImage, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Saving annotated result
    outputPath = os.path.join(OUTPUT_FOLDER, f"output_{imageName}")
    cv2.imwrite(outputPath, annotatedImage)

    # Preparing scores for CSV
    fireText = ",".join([f"{score:.4f}" for score in fireScores]) if fireScores else ""
    smokeText = ",".join([f"{score:.4f}" for score in smokeScores]) if smokeScores else ""

    # Saving results to CSV
    with open(csvFilePath, mode="a", newline="") as file:
 
        writer = csv.writer(file)
        writer.writerow([imageName, f"{fps:.2f}", fireText, smokeText])

    # Showing preview 
    cv2.imshow("Smoke and Fire Detection", annotatedImage)
    cv2.waitKey(0)

# Terminating all opened windows
cv2.destroyAllWindows()

print("Processing complete. Annotated images saved in Output_Images.")
print("Results saved in results.csv.")