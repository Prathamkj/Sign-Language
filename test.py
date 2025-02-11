import cv2  # Import OpenCV library for image processing
from cvzone.HandTrackingModule import HandDetector  # Import HandDetector from cvzone for hand tracking
from cvzone.ClassificationModule import Classifier  # Import Classifier from cvzone for gesture classification
import numpy as np  # Import numpy for array manipulations
import math  # Import math library for mathematical operations

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Initialize hand detector with a max of one hand detection
detector = HandDetector(maxHands=1)

# Load the trained model and labels for gesture classification
classifier = Classifier("C:\\Users\\User\\Desktop\\Model\\keras_model.h5", 
                        "C:\\Users\\User\\Desktop\\Model\\labels.txt")

# Set padding around the hand bounding box
offset = 20

# Set the size for resized image (used as input to the classifier)
imgSize = 300

# Initialize a counter variable (could be used for saving images or other purposes)
counter = 0

# Define labels for each class the model can classify
labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

# Start a continuous loop to process frames from the webcam
while True:
    # Capture a frame from the webcam
    success, img = cap.read()

    # If the capture fails, print an error and skip this iteration
    if not success:
        print("Failed to capture image")
        continue

    # Create a copy of the frame for output display
    imgOutput = img.copy()

    # Detect hand(s) in the frame
    hands, img = detector.findHands(img)

    # Check if any hands were detected
    if hands:
        # Select the first detected hand
        hand = hands[0]

        # Extract bounding box coordinates for the hand
        x, y, w, h = hand['bbox']

        # Ensure that the bounding box coordinates are within the frame
        if x - offset < 0 or y - offset < 0 or x + w + offset > img.shape[1] or y + h + offset > img.shape[0]:
            print("Invalid cropping coordinates:", x, y, w, h)
            continue  # Skip to the next iteration if cropping coordinates are invalid

        # Crop the image around the hand with padding
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Check if the cropped image is empty
        if imgCrop.size == 0:
            print("Cropped image is empty")
            continue  # Skip to the next iteration if empty

        # Create a blank white image for resizing hand image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Calculate aspect ratio of the hand bounding box
        aspectRatio = h / w if w > 0 else 0

        # Resize the image to fit into the white image based on aspect ratio
        if aspectRatio > 1:
            # Scale the height to the desired imgSize
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape

            # Calculate horizontal gap and place the resized hand image in the center
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            # Scale the width to the desired imgSize
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape

            # Calculate vertical gap and place the resized hand image in the center
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Make a prediction on the processed image
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        print(prediction, index)

        # Display label on the output image
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

        # Draw bounding box around the hand
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Display cropped hand image and white background image
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # Display the output image with annotations
    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)  # Wait for a key event for 1 ms before the next frame
