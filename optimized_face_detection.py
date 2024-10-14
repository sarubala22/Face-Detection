import cv2
import time

# Step 1: Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 2: Capture video from your webcam (or use a video file by specifying its path)
cap = cv2.VideoCapture(0)  # 0 is the default webcam. Change it to a path for video files

# Step 3: Set the camera resolution to reduce frame size for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480

# Optional: Set the frame rate (fps) and buffer size
cap.set(cv2.CAP_PROP_FPS, 15)  # Set frame rate to 15 fps
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce the buffer size to process frames as they arrive

# Step 4: Initialize variables for frame counting to limit detection frequency
frame_count = 0
detection_interval = 5  # Detect faces every 5 frames

# Step 5: Run a loop to continuously capture frames from the video
while True:
    # Step 6: Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Step 7: Resize frame to reduce processing time (if needed)
    frame = cv2.resize(frame, (640, 480))

    # Step 8: Increment frame counter to control detection frequency
    frame_count += 1

    if frame_count % detection_interval == 0:
        # Step 9: Convert the frame to grayscale for better detection performance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 10: Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7, minSize=(30, 30))

        # Step 11: Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Step 12: Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Step 13: Add a small delay to control frame rate and avoid overloading CPU
    time.sleep(0.05)  # Adjust this delay to control processing speed

    # Step 14: Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 15: Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

