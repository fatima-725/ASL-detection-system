import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
# vid_capture the video from the default camera
vid_cap = cv2.VideoCapture(0)
#Initialize the hand detector with a maximum of 1 hand
detector = HandDetector(maxHands=1)
# Set the offset and image size
offset = 20
image_size = 300

# Set the folder to store vid_captured images and initialize the count
folder = "Data/C"
count = 0

while True:
    # Read the frame from the camera
    success, img = vid_cap.read()
    # Find the hands in the frame using the detector
    hands, img = detector.findHands(img)
    if hands:
        # If hands are detected, get the bounding box and crop the image
        hands = hands[0]
        x, y, w, h = hands['bbox']

        # Create a white background for the cropped image
        imgWhite = np.ones((image_size, image_size, 3), np.uint8)*255
        cropped_image = img[y - offset:y + h + offset, x - offset:x + w+offset]
        # Calculate aspect ratio and resize the image accordingly
        cropped_imageShape = cropped_image.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = image_size / h
            calculated_width = math.ceil(k * w)
            resized_image = cv2.resize(cropped_image, (calculated_width , image_size))
            resized_imageShape = resized_image.shape
            wGap = math.ceil((image_size-calculated_width)/2)
            imgWhite[:, wGap:calculated_width+wGap] = resized_image

        else:
            k = image_size / w
            calculated_height = math.ceil(k * h)
            resized_image = cv2.resize(cropped_image, (image_size, calculated_height))
            resized_imageShape = resized_image.shape
            hGap = math.ceil((image_size-calculated_height)/2)
            imgWhite[hGap:calculated_height + hGap, :] = resized_image

        # Display the cropped image and the resized image with white background
        cv2.imshow("ImageCrop", cropped_image)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the original frame from the camera
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # If 's' is pressed, vid_capture the current image
    if key == ord("s"):
        count += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(count)


