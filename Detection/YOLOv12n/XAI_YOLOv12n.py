# Author: Endri Dibra 
# Project: Smoke and Fire Detection with YOLOv12n using Detection and LIME XAI

# Importing the required libraries
import os
import cv2
import numpy as np
from lime import lime_image
from ultralytics import YOLO
from skimage.segmentation import mark_boundaries


# Defining input and output paths
INPUT_FOLDER = "Input_Images"
SAVE_DIR = "XAI_Output"  

# Creating output folder for PNG saving
if not os.path.exists(SAVE_DIR):
    
    os.makedirs(SAVE_DIR)

# Loading YOLOv12n best model
model = YOLO("runs/detect/train/weights/best.pt")  

# Getting list of images in folder
imageFiles = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Checking if input folder is empty
if len(imageFiles) == 0:

    print("Error! No images found in Input_Images folder.")

    exit()

# Preparing LIME image explainer
explainer = lime_image.LimeImageExplainer()


# Helper function for LIME predictions
def yoloPredict(images):

    # List storing predictions for each LIME-perturbed image
    preds = []

    for img in images:

        # Converting the image from RGB to BGR
        imgBGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Predicting the objects/classes on the image
        result = model.predict(imgBGR, verbose=False)[0]

        # Creating probability array for all classes
        probs = np.zeros(len(model.names))

        # Sum probabilities across all detected boxes for each class
        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Calculate bounding box area
            x1, y1, x2, y2 = box.xyxy[0]
            area = max((x2 - x1) * (y2 - y1), 1)
            
            # Weighted probability by area
            probs[cls] += conf * area
        
        # Normalize probabilities to [0,1]
        if probs.sum() > 0:
            probs /= probs.sum()

        preds.append(probs)

    return np.array(preds)


# Running LIME for a given class label
def runLimeForClass(frameRGB, class_id):

    explanation = explainer.explain_instance(

        frameRGB,
        classifier_fn=yoloPredict,
        top_labels=3,
        num_samples=5000, 
        hide_color=0
    )

    temp, mask = explanation.get_image_and_mask(

        label=class_id,
        positive_only=True,
        hide_rest=False,
        num_features=100, 
        min_weight=0.05
    )

    limeImg = mark_boundaries(temp, mask)

    limeImg = (limeImg * 255).astype(np.uint8)
    limeImg = cv2.cvtColor(limeImg, cv2.COLOR_RGB2BGR)

    return limeImg, mask


# Defining class IDs for XAI
FIRE_CLASS = 0
SMOKE_CLASS = 2


# Looping through each image in Input_Images
for imageName in imageFiles:

    IMAGE_PATH = os.path.join(INPUT_FOLDER, imageName)
    IMAGE_SAVE_DIR = os.path.join(SAVE_DIR, imageName.split(".")[0])

    # Creating output folder for each image
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

    # Checking if the image file exists
    if not os.path.exists(IMAGE_PATH):

        print(f"Error! Image file does not exist: {IMAGE_PATH}")

        continue

    # Reading the image
    image = cv2.imread(IMAGE_PATH)

    # Checking if image loaded correctly
    if image is None:

        print(f"Error! Could not read image: {IMAGE_PATH}")

        continue

    print(f"Processing image: {IMAGE_PATH}")

    # Resizing orginal image to 640 x 640
    image = cv2.resize(image, (640, 640))

    # Running YOLOv12n best model detection
    results = model.predict(

        source=image,
        conf=0.5,
        imgsz=640,
        verbose=False
    )

    # Drawing YOLO detection results
    annotatedFrame = results[0].plot()

    # Saving YOLO detection output
    cv2.imwrite(f"{IMAGE_SAVE_DIR}/yoloDetection.png", annotatedFrame)

    # Converting image to RGB because LIME expects RGB format
    frameRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Running LIME for FIRE (0) and SMOKE (2)...")

    # Running LIME for fire
    limeFire, fireMask = runLimeForClass(frameRGB, FIRE_CLASS)

    # Running LIME for smoke
    limeSmoke, smokeMask = runLimeForClass(frameRGB, SMOKE_CLASS)

    # Saving separate LIME outputs
    cv2.imwrite(f"{IMAGE_SAVE_DIR}/limeFire.png", limeFire)
    cv2.imwrite(f"{IMAGE_SAVE_DIR}/limeSmoke.png", limeSmoke)

    # Creating combined heatmap
    # FIRE = RED, SMOKE = BLUE
    combined = np.zeros_like(frameRGB, dtype=np.float32)

    fireMask = fireMask.astype(np.float32)
    smokeMask = smokeMask.astype(np.float32)

    # Red channel for fire
    combined[:, :, 2] += fireMask * 255

    # Blue channel for smoke
    combined[:, :, 0] += smokeMask * 255

    # Mixing heatmap with the original image
    alpha = 0.5

    # Combining the two masks of Fire and Smoke into one 
    combinedOverlay = cv2.addWeighted(

        frameRGB.astype(np.uint8),
        1 - alpha,
        combined.astype(np.uint8),
        alpha,
        0
    )

    combinedOverlay = cv2.cvtColor(combinedOverlay, cv2.COLOR_RGB2BGR)

    # Saving combined heatmap
    cv2.imwrite(f"{IMAGE_SAVE_DIR}/limeCombined.png", combinedOverlay)

    print("Detection and LIME XAI complete!")
    print(f"Results saved in folder: {IMAGE_SAVE_DIR}")
    print("Press any key to close windows.")


# Closing all opened windows
cv2.destroyAllWindows()