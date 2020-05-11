#-*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import cv2

from tensorflow import keras
import tensorflowjs as tfjs
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet, InceptionV3, Xception, InceptionResNetV2, NASNetMobile
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing import image

# TRAINING_DATA_SET_PATH = '/content/SSL-Dataset/train'
# VALIDATION_DATA_SET_PATH = '/content/SSL-Dataset/test'
#TEST_DATA_SET_PATH = '../data/test-images/A1.jpg'
#LOGS_PATH = '/content/logs'
EPOCHS = 10
BATCH_SIZE = 32
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
DATASET_CATEGORIES = 26

# train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
#                                    rotation_range=13,
#                                    width_shift_range=0.3,
#                                    height_shift_range=0.3,
#                                    shear_range=0.3,
#                                    zoom_range=0.4,
#                                    fill_mode='nearest'
#                                    )
#
# val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#
# train_generator = train_datagen.flow_from_directory(TRAINING_DATA_SET_PATH,
#                                                     target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
#                                                     color_mode='rgb',
#                                                     batch_size=BATCH_SIZE,
#                                                     class_mode='categorical',
#                                                     shuffle=True)
#
# validation_generator = val_datagen.flow_from_directory(
#     VALIDATION_DATA_SET_PATH,
#     target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical')

# test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# test_generator = test_datagen.flow_from_directory(
#     TEST_DATA_SET_PATH,
#     target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical')

def getTopPredictions(preds):

    predsDict = {
        'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0,
        'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
        'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0,
        'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0
    }
    # map preds index with probability to correct letter from dictonary
    for i, key in enumerate(predsDict, start=0):
        predsDict[key] = preds[i]

    # sort by dictonary value and returns as list
    all_preds = sorted(predsDict.items(), reverse=True, key=lambda x: x[1])

    top_preds = [all_preds[0], all_preds[1], all_preds[2]]

    return top_preds, all_preds

def predictSingleImage(filepath, model):
    inputImage = Image.open(filepath)
    inputImage = inputImage.resize((224, 224))

    imageDataArray = image.img_to_array(inputImage)
    imageDataArray = np.expand_dims(imageDataArray, axis=0)
    imageDataArray = preprocess_input(imageDataArray)

    predictions = model.predict(imageDataArray)
    return predictions

def loadModelFromFile(modelPath):
    print("Loading model...", modelPath)
    loaded_model = keras.models.load_model(modelPath)
    print("Model loaded from", modelPath)
    return loaded_model

# def evaluateModel(model):
#
#     model.compile(optimizer='Adam', loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     score = model.evaluate_generator(validation_generator,
#                                      validation_generator.n / BATCH_SIZE,
#                                      verbose=1)
#     loss, accuracy = score[0], score[1]
#     return loss, accuracy

def showImageCV(imagePath, top_three_preds):
    #bestGuess = next(iter(top_three_preds))
    letter, accuracy = next(iter(top_three_preds))
    text = "{}: {:.2f}%".format(letter, accuracy * 100)

    image = cv2.imread(imagePath)
    output = image.copy()
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,255,4), 2)
    # show the output image
    cv2.imshow("Image", output)
    cv2.waitKey(0)


#Method for streaming video from webcam 
def videoStream(finalModel):
    #Set source to webcam
    capture = cv2.VideoCapture(0)
    predictionsBuffer = []
    wordBuilder = []
    startRecording = False

    #continuous stream
    while(True):
        #Set the width and height to 240*240
        ret = capture.set(3,240)
        ret = capture.set(4,240)

        # Capture frame-by-frame
        ret, frame = capture.read()

        top_three_preds, all_preds = getTopPredictions(predictSingleImage('singleFrame.png', finalModel)[0])
        #print('The letter is:', top_three_preds[0])
        letter, accuracy = top_three_preds[0]

        if len(predictionsBuffer) >= 8 and startRecording:
            predictionsBuffer = predictionsBuffer[1:]
            predictionsBuffer.append(letter)
        elif startRecording:
            predictionsBuffer.append(letter)
        if len(predictionsBuffer) >= 8:
            isEqual = all(elem == predictionsBuffer[0] for elem in predictionsBuffer)
            lastLetter = len(wordBuilder) <= 0 and '-' or wordBuilder[-1]
            if isEqual and (letter != lastLetter) and startRecording:
                wordBuilder.append(letter)
                print('Added', letter)
                cv2.putText(frame, 'Added ' + letter, (300, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,255,4), 2)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('s'):
            startRecording = True
            print('Started recording. Start signing.')
            cv2.putText(frame, 'Started recording. Start signing.', (140, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,4), 2)
        if keypress == 32:
            print('Added space')
            wordBuilder.append(' ')
            cv2.putText(frame, 'Added space', (300, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,4), 2)
        if keypress == ord('d'):
            wordBuilder = wordBuilder[:-1]
            cv2.putText(frame, 'Delete last letter', (300, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,4), 2)
        if keypress == ord('r'):
            cv2.putText(frame, 'Reset', (300, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,4), 2)
            wordBuilder = []
        if keypress == ord('q') or keypress == 27:
            break

        if startRecording:
            print(''.join(wordBuilder))
        else:

            cv2.putText(frame, 'Press S on keyboard to start interpreting.', (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,4), 2)
            print('Press S on keyboard to start interpreting.')

        text = "{} with {:.2f}% confidence".format(letter, accuracy * 100)
        cv2.putText(frame, text, (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0,255,4), 2)
        cv2.putText(frame, (''.join(wordBuilder)), (210, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0,255,4), 2)
        cv2.putText(frame, 'Press:', (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,4), 2)
        cv2.putText(frame, 'd to delete', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,4), 2)
        cv2.putText(frame, 'space for space', (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,4), 2)
        cv2.putText(frame, 'r to reset', (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,4), 2)
        cv2.putText(frame, 'q to quit', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,4), 2)


        cv2.imshow('SignInterpreterSSL', frame)
        # cv2.waitKey(30)

        cv2.imwrite('singleFrame.png', frame)

    # Release the capture
    capture.release()
    cv2.destroyAllWindows()
    print('Exit Program')


def mainPipeline():
    
    finalModel = loadModelFromFile('../../best-models/dark-snowball-202.h5')
    videoStream(finalModel)
    #imagePath = '../data/test-images/G/G.jpg'
    #predictions = predictSingleImage(imagePath, finalModel)[0]

    #print('*****************************************************')
    #loss, accuracy = evaluateModel(finalModel)
    #print("Loss: ", loss, "Accuracy: ", accuracy * 100, '%')
    #print('Input was:', letter)
    #top_three_preds, all_preds = getTopPredictions(predictions)
    #print(top_three_preds)
    #print(finalModel.summary())

    #showImageCV(imagePath, top_three_preds)

mainPipeline()