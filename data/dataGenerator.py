#-*- coding: utf-8 -*-
# Importing all necessary libraries
import cv2
import os


categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z', 'Å', 'Ä', 'Ö']


# categories = ['A']

def createImagesFromClip(letter):
    folderPath = 'clips/' + letter + '/'

    for filename in os.listdir(folderPath):
        clipPath = os.path.join(folderPath, filename)
    # Read the video from specified path
    videoStream = cv2.VideoCapture(clipPath)

    currentframe = 1
    counter = 0
    while True:

        live, frame = videoStream.read()
        everyOther = True

        if live and currentframe <= 600:  # Continue as long as video still has frames

            if currentframe <= 500: #Train/Test split
                folder = 'train'
            else:
                folder = 'test'
            try:  # Create a folder if it doesnt exists
                dataFolder = 'data/SSL-dataset/' + folder + '/' + letter
                if not os.path.exists(dataFolder):
                    os.makedirs(dataFolder)


            except OSError:
                pass

            name = './data/SSL-dataset/' + folder + '/' + letter + '/' + letter + str(currentframe) + '.jpg'
            print('Creating...' + name)

            #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Crop image. First is height
            frame = frame[0:1080, 420:1500]
            frame = cv2.resize(frame, (224, 224))

            cv2.imwrite(name, frame)

            # counter += 15  # Capture every 15th frame (if 30fps, every 0.5 seconds of clip
            # videoStream.set(1, counter)

            currentframe += 1

        else:
            break

    # Release all space and windows once done
    videoStream.release()
    cv2.destroyAllWindows()


for letter in categories:
    createImagesFromClip(letter)
    print('Done with', letter)