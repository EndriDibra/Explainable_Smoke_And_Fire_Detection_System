# Author: Endri Dibra 
# Project: Hybrid Smoke and Fire Detection using Sensor Fusion [Sensor data reading + Vision-Detection]

# Importing the required libraries 
import cv2
import time
import pandas as pd
from joblib import load
from ultralytics import YOLO


# Defining input CSV file for sensor readings
inputSensorCsv = "C:/Users/User/Documents/AI_Robotics Projects/Smoke_Fire_Sensor_Fusion/Processing/Dataset.csv"

# Below i am invoking the best tabular and detection model
# Based on the comparison made for each case

# Defining path to YOLO best trained model weights
yoloModelPath = "C:/Users/User/Documents/AI_Robotics Projects/Smoke_Fire_Sensor_Fusion/Detection/YOLOv5nu/runs/detect/train/weights/best.pt"

# Defining path to best pre-trained Tabular sensor ML model
sensorModelPath = "C:/Users/User/Documents/AI_Robotics Projects/Smoke_Fire_Sensor_Fusion/Processing/Models/best_AutoML_model_RandomForest.joblib"

# Setting flag for saving explainable AI outputs
saveXai = False

# Setting flag for displaying FPS on video
fpsDisplay = True

# Setting fusion weight for tabular sensor model
wTabular = 0.55

# Setting fusion weight for YOLO detection model
wDetection = 0.45

# Fire: Higher bars for immediate threat
# Minimum fire probability threshold
fireConfirmThreshold = 0.65

# Minimum sensor probability threshold (double security)
fireSensorMinThreshold = 0.65

# Minimum fire probability threshold for override cases
# Where there is a fire detected, but it is far
# from the radius distance that sensors can read the danger
fireYoloOverrideThreshold = 0.85

# Smoke: Lower bars for early/ambient detection
# Minimum smoke probability threshold
smokeConfirmThreshold = 0.65

# Minimum sensor probability threshold (double security)
smokeSensorMinThreshold = 0.65 

# Minimum smoke probability threshold for override cases
# Where there is a smoke detected, but it is far
# from the radius distance that sensors can read the danger
smokeYoloOverrideThreshold = 0.85

# Boundaries for the Fusion Dataset: Limited to 10000 rows
# 5000 rows from the sensor dataset, from start (row 0): Class 0, No Fire Alarm
# 5000 rows from the sensor dataset, from row 3178: Class 1, Fire Alarm
maxRows = 10000
startFireRows = 3178

# Output Fusion Dataset CSV
# Saving sensor data
# Plus smoke and fire label alarms from YOLO detections
# To train a new Neural Network on them
# To obtain a Sensor Fused trained NN model 
fusionDataset = "FusionDataset.csv"

# Loading best YOLO model for visual detection
yoloModel = YOLO(yoloModelPath)

# Loading best pre-trained tabular sensor ML model
sensorModel = load(sensorModelPath)


# Defining function for weighted product fusion of sensor and YOLO probabilities
def weightedProductFusion(pTabular, pDetection, wTabular=wTabular, wDetection=wDetection):
    
    # Returning combined probability using weighted product
    return (pTabular ** wTabular) * (pDetection ** wDetection)


# Defining function to separate fusion for fire alert
def fireAlert(pTabular, pFire, thresholdConfirm=fireConfirmThreshold, sensorMinThreshold=fireSensorMinThreshold, yoloOverrideThreshold=fireYoloOverrideThreshold):
    
    # Defining the fire result from the weighted product fusion
    sConfirm = weightedProductFusion(pTabular, pFire, wTabular, wDetection)
    
    # Checking if the current fire probability is less than the defined threshold
    if sConfirm < thresholdConfirm:
        
        # Fire override case
        if pTabular >= sensorMinThreshold and pFire >= yoloOverrideThreshold:
   
            return 1
   
        return 0
   
    return int(sConfirm >= thresholdConfirm)


# Defining function to separate fusion for smoke alert
def smokeAlert(pTabular, pSmoke, thresholdConfirm=smokeConfirmThreshold, sensorMinThreshold=smokeSensorMinThreshold, yoloOverrideThreshold=smokeYoloOverrideThreshold):
    
    # Defining the smoke result from the weighted product fusion
    sConfirm = weightedProductFusion(pTabular, pSmoke, wTabular, wDetection)
    
    # Checking if the current smoke probability is less than the defined threshold
    if sConfirm < thresholdConfirm:
        
        # Smoke override case
        if pTabular >= sensorMinThreshold and pSmoke >= yoloOverrideThreshold:
    
            return 1
    
        return 0
    
    return int(sConfirm >= thresholdConfirm)


# Defining function to return rows' feature readings
def preprocessSensor(sensorRow):

    # Copying row to avoid modifying the original one
    features = sensorRow.copy()

    # Returning rows' processed features as raw DataFrame
    return features

# Defining function to preprocess input dataset
def preprocessInputSensorData(df):
    
    # Removing duplicate rows
    df = df.drop_duplicates()

    # Filling missing numeric values with column median
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df


# Defining function to preprocess Fusion Dataset
def cleanFusionDataset(fusionDf):
   
    # Removing duplicate rows
    fusionDf = fusionDf.drop_duplicates()

    # Filling missing numeric values with column median
    numericCols = fusionDf.select_dtypes(include=['float64', 'int64']).columns
    fusionDf[numericCols] = fusionDf[numericCols].fillna(fusionDf[numericCols].median())

    return fusionDf


# Defining function to extract separately YOLO confidence for fire and smoke
def getYoloConfidence(frame):

    # Predicting objects using YOLO model
    results = yoloModel.predict(source=frame, conf=0.5, imgsz=640, verbose=False)
    boxes = results[0].boxes
    
    # Class 0: Fire
    pFire = 0.0 

    # Class 2: Smoke
    pSmoke = 0.0  

    # Iterating through detected boxes
    for box in boxes:
        
        # Class ID
        clsId = int(box.cls[0])

        # Confidence score
        conf = float(box.conf[0])

        # Separating max confidence for fire and smoke
        if clsId == 0:
            
            # Taking max confidence score for fire case
            pFire = max(pFire, conf)
        
        elif clsId == 2:

            # Taking max confidence score for smoke case
            pSmoke = max(pSmoke, conf)

    # Taking the max confidence from fire vs smoke
    # For display/legacy (max of both)
    pDetectionCombined = max(pFire, pSmoke)

    # Returning separate probs, combined, and annotated frame
    return pFire, pSmoke, pDetectionCombined, results[0].plot()


# Reading sensor's dataset CSV into dataframe
dataFrame = pd.read_csv(inputSensorCsv)

# Preprocessing input dataset
dataFrame = preprocessInputSensorData(dataFrame)

# Renaming columns for simplicity
dataFrame.rename(columns={
  
    "Humidity[%]": "Humidity",
    "Temperature[C]": "Temperature",
    "TVOC[ppb]": "TVOC",
    "eCO2[ppm]": "ECO2",
    "Pressure[hPa]": "Pressure"

}, inplace=True)

# Dropping unnecessary columns
dropCols = ["Unnamed: 0", "CNT", "UTC"]
dataFrame.drop(columns=dropCols, inplace=True, errors='ignore')

# Splitting features and target variable
x = dataFrame.drop("Fire Alarm", axis=1)
y = dataFrame["Fire Alarm"]

# Selecting 5000 rows from start (class 0: No Fire Alarm)
# And 5000 rows from row 3178 (class 1: Fire Alarm)
xLimited1 = x.iloc[0:5000]
xLimited2 = x.iloc[startFireRows:startFireRows + 5000] if len(x) >= startFireRows + 5000 else x.iloc[startFireRows:]
xLimited = pd.concat([xLimited1, xLimited2])
yLimited = pd.concat([y.iloc[0:5000], y.iloc[startFireRows:startFireRows + 5000]]) if len(y) >= startFireRows + 5000 else pd.concat([y.iloc[0:5000], y.iloc[startFireRows:]])

print(f"Dataset loaded: Total rows {len(xLimited)}, Fire alarms: {yLimited.sum()}")

# Preparing list to collect fusion data
fusionData = []

# Headers to include separate fire/smoke alerts
headers = ['Temperature', 'Humidity', 'TVOC', 'ECO2', 'Raw H2', 'Raw Ethanol', 'Pressure', 
           'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'Ptabular', 'Pfire', 'Psmoke', 'Pdetection', 
           'FireAlert', 'SmokeAlert']

# Opening the defaul web camera 
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Checking if camera opened correctly
if not camera.isOpened():

    print("Error! Camera did not open.")

    exit()

# Initializing previous time for FPS calculation
prevTime = 0

# Sensor row index for cycling through 10000 rows
sensorIdx = 0

# Starting continuous loop for camera stream, limited to 10000 frames/rows
frameCount = 0

while camera.isOpened() and frameCount < maxRows:
  
    # Reading frame from camera
    success, frame = camera.read()
  
    # Checking if frame was captured successfully
    if not success:
  
        print("Error! While reading camera frame.")
  
        break

    # Cycling through sensor rows for variety
    currentRow = xLimited.iloc[sensorIdx].to_frame().T

    # Cycling back if needed
    sensorIdx = (sensorIdx + 1) % len(xLimited)  

    # Preprocessing and predicting pTabular for this sensor row
    sensorFeatures = preprocessSensor(currentRow)
    pTabular = sensorModel.predict_proba(sensorFeatures)[0, 1]

    # Predicting separate fire/smoke probabilities from YOLO model
    pFire, pSmoke, pDetection, annotatedFrame = getYoloConfidence(frame)

    # Separating alerts for fire and smoke
    fireAlertStatus = fireAlert(pTabular, pFire, fireConfirmThreshold, fireSensorMinThreshold, fireYoloOverrideThreshold)
    smokeAlertStatus = smokeAlert(pTabular, pSmoke, smokeConfirmThreshold, smokeSensorMinThreshold, smokeYoloOverrideThreshold)

    # Collecting data for fusion dataset, with 12 sensor features
    sensorVals = currentRow.iloc[0].values.tolist()  
    fusionRow = sensorVals + [pTabular, pFire, pSmoke, pDetection, fireAlertStatus, smokeAlertStatus]
    fusionData.append(fusionRow)

    # Calculating FPS
    currTime = time.time()
    fps = 1 / (currTime - prevTime) if prevTime != 0 else 0
    prevTime = currTime

    # Displaying FPS on annotated frame
    if fpsDisplay:
  
        cv2.putText(annotatedFrame, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    # Displaying sensor probability
    cv2.putText(annotatedFrame, f"Sensor Prob: {pTabular:.2f}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)

    # Displaying separate fire/smoke probabilities
    cv2.putText(annotatedFrame, f"Fire Prob: {pFire:.2f}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
    cv2.putText(annotatedFrame, f"Smoke Prob: {pSmoke:.2f}", (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)

    # Displaying combined YOLO probability (legacy)
    # cv2.putText(annotatedFrame, f"YOLO Combined: {pDetection:.2f}", (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

    # Displaying separate alert statuses
    cv2.putText(annotatedFrame, f"Fire Alert: {'YES' if fireAlertStatus else 'NO'}", (10,190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if fireAlertStatus else (0,255,0),2)
    cv2.putText(annotatedFrame, f"Smoke Alert: {'YES' if smokeAlertStatus else 'NO'}", (10,220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if smokeAlertStatus else (0,255,0),2)

    # Showing annotated frame
    cv2.imshow("Hybrid Smoke and Fire Detection", annotatedFrame)

    # Breaking loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
  
        break

    frameCount += 1

# Releasing camera resources
camera.release()

# Closing all OpenCV windows
cv2.destroyAllWindows()


# Cleaning and saving Fusion Dataset to CSV
fusionDf = pd.DataFrame(fusionData, columns=headers)
fusionDf = cleanFusionDataset(fusionDf)
fusionDf.to_csv(fusionDataset, index=False)

print(f"FusionDataset.csv created with {len(fusionData)} rows.") 