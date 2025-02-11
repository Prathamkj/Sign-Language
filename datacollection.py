import cv2  # Import OpenCV library for image and video processing
from cvzone.HandTrackingModule import HandDetector  # Import HandDetector from cvzone for hand tracking
import numpy as np  # Import numpy library for array manipulations
import math  # Import math library for mathematical operations
import time  # Import time library for time-based functions

cap = cv2.VideoCapture(0)  # Initialize video capture with the default camera (index 0)
detector = HandDetector(maxHands=1)  # Create a hand detector allowing a maximum of one hand detection
offset = 20  # Set padding offset for cropping the hand region
imgSize = 300  # Define the size of the white background image for resizing
counter = 0  # Initialize a counter for saving images

folder = "C:\\Users\\User\\Desktop\\Hand-Gesture\\Data\\Yes"  # Path to save captured images

while True:
    success, img = cap.read()  # Capture each frame from the webcam
    hands, img = detector.findHands(img)  # Detect hands in the frame, updating img with hand landmarks
    if hands:  # If a hand is detected
        hand = hands[0]  # Get the first detected hand
        x, y, w, h = hand['bbox']  # Extract bounding box coordinates of the detected hand

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Create a white background of imgSize x imgSize

        # Calculate crop coordinates within image boundaries to avoid errors
        x1, y1 = max(0, x - offset), max(0, y - offset)  # Top-left corner with offset
        x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)  # Bottom-right corner with offset
        imgCrop = img[y1:y2, x1:x2]  # Crop the image around the detected hand

        if imgCrop.size == 0:  # If the cropped image is empty, skip resizing
            print("imgCrop is empty, skipping resize.")
            continue  # Continue to the next loop iteration

        imgCropShape = imgCrop.shape  # Get shape of the cropped image
        aspectRatio = h / w  # Calculate the aspect ratio of the hand bounding box

        if aspectRatio > 1:  # If height is greater than width (tall aspect ratio)
            k = imgSize / h  # Scaling factor based on height
            wCal = math.ceil(k * w)  # Calculate new width while maintaining aspect ratio
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Resize the cropped image based on calculated width
            wGap = math.ceil((imgSize - wCal) / 2)  # Calculate gap to center image horizontally
            imgWhite[:, wGap:wCal + wGap] = imgResize  # Place the resized image on the white background
        else:  # If width is greater than or equal to height (wide aspect ratio)
            k = imgSize / w  # Scaling factor based on width
            hCal = math.ceil(k * h)  # Calculate new height while maintaining aspect ratio
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Resize the cropped image based on calculated height
            hGap = math.ceil((imgSize - hCal) / 2)  # Calculate gap to center image vertically
            imgWhite[hGap:hCal + hGap, :] = imgResize  # Place the resized image on the white background

        cv2.imshow('ImageCrop', imgCrop)  # Display the cropped hand image
        cv2.imshow('ImageWhite', imgWhite)  # Display the white background image with the hand

    cv2.imshow('Image', img)  # Display the original frame from the webcam
    key = cv2.waitKey(1)  # Wait for a key press with a delay of 1 ms
    if key == ord("s"):  # If 's' key is pressed
        counter += 1  # Increment the image counter
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)  # Save the white background image to the folder
        print(counter)  # Print the current counter value
