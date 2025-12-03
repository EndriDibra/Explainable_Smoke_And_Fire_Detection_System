# Author: Endri Dibra 
# Project: Smoke and Fire Detection with YOLOv5nu using Detection

# Importing the required libraries
import cv2
import time
from ultralytics import YOLO


# Loading YOLOv5nu best model
model = YOLO("runs/detect/train/weights/best.pt")  

# Opening the default camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

# Checking if camera works 
if not camera.isOpened():
    
    print("Error! Camera did not open.")
    
    exit()

# Initialize previous time for FPS calculation
prevTime = 0

# Looping through camera frames
while camera.isOpened():

    # Reading camera frames
    success, frame = camera.read()
    
    # Checking if camera frames work
    if not success:
    
        print("Error! Camera frames did not open.")
    
        break

    # Calculate FPS
    currTime = time.time()
    fps = 1 / (currTime - prevTime) if prevTime != 0 else 0
    prevTime = currTime

    # Running YOLOv5nu best model detection
    results = model.predict(
    
        source=frame,
        conf=0.5,      
        imgsz=640,
        verbose=False
    )

    # Drawing results on the frame
    # YOLO boxes and labels
    annotatedFrame = results[0].plot()  

    # Add FPS text to the frame
    cv2.putText(annotatedFrame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Smoke and Fire Detection", annotatedFrame)

    # Terminating program by pressing key "q":quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
    
        break


# Closing all opened windows
camera.release()
cv2.destroyAllWindows()