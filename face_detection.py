# OpenCV Face Detection using Haar Cascade


import cv2
import cv2.data

# Load the Haar cascade for face detection
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture
video_cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not video_cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, video = video_cap.read()
    if not ret:
        break

    # Convert to grayscale
    col = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the video
    cv2.imshow("Video_live", video)

    # Exit when 'q' is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release the capture and destroy windows
video_cap.release()
cv2.destroyAllWindows()
