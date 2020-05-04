#-*- coding: utf-8 -*-
# Importing all necessary libraries
import cv2
import os


categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']

datasetIDsFirst = ['1G', '2G', '1J', '2J', '3J']
datasetIDsSecond = ['1K', '1P', '1S']

def createImagesFromClip(letter, datasetID):
    folderPath = 'clips/' + datasetID + '/' + letter + '/'

    for filename in os.listdir(folderPath):
        clipPath = os.path.join(folderPath, filename)
    # Read the video from specified path
    videoStream = cv2.VideoCapture(clipPath)

    currentframe = 1
    counter = 0
    while True:
        live, frame = videoStream.read()
        if live and currentframe <= 200:  # Continue as long as video still has frames

            if currentframe <= 100: #Train/Test/Validation split
                folder = 'train'
            elif currentframe <= 150 :
                folder = 'validation'
            else:
                folder = 'test'
            try:  # Create a folder if it doesnt exists
                dataFolder = 'data/SSL-dataset/' + folder + '/' + letter
                if not os.path.exists(dataFolder):
                    os.makedirs(dataFolder)


            except OSError:
                pass

            name = './data/SSL-dataset/' + folder + '/' + letter + '/' + letter + datasetID + str(currentframe) + '.jpg'
            print('Creating...' + name)

            #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Crop image. First is height
            frame = frame[0:1080, 200:960]
            frame = cv2.resize(frame, (224, 224))

            cv2.imwrite(name, frame)

            counter += 3  # Capture every 3th frame (if 30fps, every 0.1 seconds of clip
            videoStream.set(1, counter)

            currentframe += 1

        else:
            break

    # Release all space and windows once done
    videoStream.release()
    cv2.destroyAllWindows()


for id in datasetIDsFirst:
    for letter in categories:
        createImagesFromClip(letter, id)
        print('Done with', letter)
    print('Done with', id)