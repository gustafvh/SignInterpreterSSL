# https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e

import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
from skimage import transform

from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import model_from_json

import cv2


#### Variables ######

TRAINING_DATA_SET_PATH = './data/asl-alphabet/asl_alphabet_train/asl_alphabet_train'



def getPreTrainedModel():
    base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(3,activation='softmax')(x) #final layer with softmax activation
    model=Model(inputs=base_model.input,outputs=preds)
    return model

def freezeLayers(numOfLayers, model):
    for layer in model.layers[:numOfLayers]:
        layer.trainable=False
    for layer in model.layers[numOfLayers:]:
        layer.trainable=True

    return model

def fitModel(model):
    train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

    train_generator=train_datagen.flow_from_directory(TRAINING_DATA_SET_PATH, # this is where you specify the path to the main data folder
                                                      target_size=(224,224),
                                                      color_mode='rgb',
                                                      batch_size=32,
                                                      class_mode='categorical',
                                                      shuffle=True)


    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy

    step_size_train=train_generator.n//train_generator.batch_size
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_train,
                        epochs=5)
    return model

def saveModelToJson(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("./output/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./output/model.h5")
    print("Saved model to disk")

def loadModelfromJson(modelPath, weightPath):
    # load json and create model
    json_file = open(modelPath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weightPath)
    print("Loaded model from disk")
    return loaded_model

def loadSingleImage(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def trainNetwork():
    preTrainedModel = getPreTrainedModel()
    trimmedModel = freezeLayers(20, preTrainedModel)
    finalModel = fitModel(trimmedModel)
    saveModelToJson(finalModel)
    return finalModel


def showImageCV(preds):
    # find the class label index with the largest corresponding
    # probability
    i = preds.argmax(axis=1)[0]
    label = 'C'
    # label = lb.classes_[i]
    # draw the class label + probability on the output image
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)

    image = cv2.imread('./data/asl-alphabet/asl_alphabet_train/asl_alphabet_train/C/C1.jpg')
    output = image.copy()
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)
    # show the output image
    cv2.imshow("Image", output)
    cv2.waitKey(0)

def mainPipeline():
    #finalModel = trainNetwork()
    finalModel = loadModelfromJson('./output/model.json', './output/model.h5')
    image = loadSingleImage('./data/asl-alphabet/asl_alphabet_train/asl_alphabet_train/C/C1.jpg')
    predictions = finalModel.predict(image)
    # Predictions contains a 2D array where the index of the
    # images sent in (so for a single image its only a
    # prediction[0]) and in that array is the prediction score for
    # each class A,B,C
    print(predictions)
    print(predictions.astype('int'))
    showImageCV(predictions)


mainPipeline()