# Importing all necessary libraries
import cv2
import os

# Read the video from specified path
videoStream = cv2.VideoCapture("clips/G/G001.MOV")

currentframe = 1
counter = 0

while (True):

    ret, frame = videoStream.read()
    everyOther = True

    if ret:
        # if video is still left continue creating images
        name = './data/SSL-dataset/train/G/G' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        frame = cv2.flip(frame, -1)
        cv2.imwrite(name, frame)

        counter += 15 #Capture every 15th frame (if 30fps, every 0.5 seconds of clip
        videoStream.set(1, counter)

        currentframe += 1

    else:
        break

# Release all space and windows once done
videoStream.release()
cv2.destroyAllWindows()
