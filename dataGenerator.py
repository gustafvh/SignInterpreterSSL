# Importing all necessary libraries
import cv2
import os

# categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
#               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#               'V', 'W', 'X', 'Y', 'Z', 'Å', 'Ä', 'Ö']

categories = ['G']


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

        if live:  # Continue as long as video still has frames

            try:
                # creating a folder named data
                dataFolder = 'data/SSL-dataset/train/' + letter
                if not os.path.exists(dataFolder):
                    os.makedirs(dataFolder)

                # if not created then raise error
            except OSError:
                pass

            name = './data/SSL-dataset/train/' + letter + '/' + letter + str(currentframe) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            frame = cv2.flip(frame, -1)

            frame = cv2.resize(frame, (224, 224))

            cv2.imwrite(name, frame)

            counter += 15  # Capture every 15th frame (if 30fps, every 0.5 seconds of clip
            videoStream.set(1, counter)

            currentframe += 1

        else:
            break

    # Release all space and windows once done
    videoStream.release()
    cv2.destroyAllWindows()


for letter in categories:
    createImagesFromClip(letter)
