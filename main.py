from PIL import Image
import numpy as np
import pandas as pd
from skimage import transform

from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import model_from_json

import cv2

from keras.datasets import mnist

#### Variables ######

TRAINING_DATA_SET_PATH = './data/asl-alphabet-small/asl_alphabet_train'
VALIDATION_DATA_SET_PATH = './data/asl-alphabet-small/asl_alphabet_test'
EPOCHS = 3
BATCH_SIZE = 32
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
DATASET_CATEGORIES = 3


####################

def getPreTrainedModel():
    pre_trained_base_model = MobileNet(weights='imagenet',
                                       include_top=False)  # Remove last neuron layer (that contains 1000 neurons)

    my_model = pre_trained_base_model.output
    my_model = GlobalAveragePooling2D()(my_model)
    my_model = Dense(1024, activation='relu')(
        my_model)  # Add layer. Layer #1: 1000 nodes, and rectified linear activation function
    my_model = Dense(1024, activation='relu')(my_model)  # Layer #2
    my_model = Dense(512, activation='relu')(my_model)
    final_layer = Dense(DATASET_CATEGORIES, activation='softmax')(my_model)  # DATASET_CATEGORIES because it could be A,B,C etc.
    final_model = Model(inputs=pre_trained_base_model.input, outputs=final_layer)
    return final_model


def freezeLayers(numOfLayers, model):
    for single_layer in model.layers[:numOfLayers]:
        single_layer.trainable = False
    for single_layer in model.layers[numOfLayers:]:
        single_layer.trainable = True

    return model


def fitModel(model):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       preprocessing_function=preprocess_input)  # For training data

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(TRAINING_DATA_SET_PATH,
                                                        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                        color_mode='rgb',
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical',
                                                        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DATA_SET_PATH,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    step_size_training = train_generator.n // train_generator.batch_size  # Balance here to find optimal step_size
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_training,
                        epochs=EPOCHS,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.n // BATCH_SIZE)

    return model


def trainNetwork():
    preTrainedModel = getPreTrainedModel()
    trimmedModel = freezeLayers(20, preTrainedModel)
    finalModel = fitModel(trimmedModel)
    saveModelToJson(finalModel)
    return finalModel


def saveModelToJson(model):
    model_json = model.to_json()
    with open("./output/model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("./output/model.h5")
    print("Model saved as ./output/model.h5")


def loadModelfromJson(modelPath, weightPath):
    json_file = open(modelPath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(weightPath)
    print("Model loaded from", modelPath)
    return loaded_model


def loadSingleImage(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


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


def getTopPredictions(preds):
    # top_preds = preds.argmax(axis=1)[0]

    # Convert from float into percentages
    top_preds = pd.DataFrame(preds / float(np.sum(preds))).applymap(lambda preds: '{:.2%}'.format(preds)).values

    return top_preds


def evaluateModel(model):
    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      preprocessing_function=preprocess_input)

    validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DATA_SET_PATH,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    score = model.evaluate_generator(validation_generator, validation_generator.n / BATCH_SIZE, workers=12)

    print("Loss: ", score[0], "Accuracy: ", score[1]*100, '%')


def mainPipeline():
    finalModel = trainNetwork()
    #finalModel = loadModelfromJson('./output/model.json', './output/model.h5')

    image = loadSingleImage('./data/asl-alphabet/asl_alphabet_train/C/C125.jpg')
    predictions = finalModel.predict(image)
    # Predictions contains a 2D array where the index of the
    # images sent in (so for a single image its only a
    # prediction[0]) and in that array is the prediction score for
    # each class A,B,C
    # user a dataframe to format the numbers and then convert back to a numpy array.
    print('*****************************************************')
    print('Predictions that its [Class A, Class B, ....]')
    # print(predictions)
    print(getTopPredictions(predictions))
    # print(predictions)
    # showImageCV(predictions)

    print('*****************************************************')
    print(evaluateModel(finalModel))


mainPipeline()

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(np.shape(X_train))
# print(type(X_train))
