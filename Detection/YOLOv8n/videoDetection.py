# Author: Endri Dibra 
# Project: Smoke and Fire Detection with YOLOv8n using Detection

# Importing the required libraries
import cv2
import time
from ultralytics import YOLO


# Loading YOLOv8n best model
model = YOLO("runs/detect/train/weights/best.pt")  

# Opening the MP4 video file
video_path = "fire1.mp4"
cap = cv2.VideoCapture(video_path)

# Checking if video works 
if not cap.isOpened():
   
    print("Error! Video did not open.")
   
    exit()

# Getting video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Defining codec and create VideoWriter object to save output
out = cv2.VideoWriter(
    
    'fire1_detected.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

# Initializing previous time for FPS calculation
prevTime = 0

# Looping through video frames
while cap.isOpened():
    
    success, frame = cap.read()
    
    if not success:
    
        print("End of video or error reading frame.")
        break

    # Calculating FPS
    currTime = time.time()
    fps_display = 1 / (currTime - prevTime) if prevTime != 0 else 0
    prevTime = currTime

    # Running YOLOv8n model detection
    results = model.predict(

        source=frame,
        conf=0.5,
        imgsz=640,
        verbose=False
    )

    # Drawing results on the frame
    annotatedFrame = results[0].plot()

    # Add FPS text to the frame
    cv2.putText(annotatedFrame, f"FPS: {fps_display:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the annotated frame to the output video
    out.write(annotatedFrame)

    # Displaying the frame
    cv2.imshow("Smoke and Fire Detection", annotatedFrame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
    
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved as 'fire1_detected.mp4'.")