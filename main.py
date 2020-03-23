from PIL import Image
import numpy as np
from skimage import transform

from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import model_from_json

import cv2

#### Variables ######

TRAINING_DATA_SET_PATH = './data/asl-alphabet/asl_alphabet_train/asl_alphabet_train'
EPOCHS = 5
BATCH_SIZE = 32
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


####################

def getPreTrainedModel():
    pre_trained_base_model = MobileNet(weights='imagenet',
                                       include_top=False)  # Remove last neuron layer (that contains 1000 neurons)

    my_model = pre_trained_base_model.output
    my_model = GlobalAveragePooling2D()(my_model)
    my_model = Dense(1000, activation='relu')(
        my_model)  # Add layer. Layer #1: 1024 nodes, and rectified linear activation function
    my_model = Dense(1000, activation='relu')(my_model)  # Layer #2
    my_model = Dense(500, activation='relu')(my_model)
    final_layer = Dense(3, activation='softmax')(my_model) # 3 because it could be A,B or C
    final_model = Model(inputs=pre_trained_base_model.input, outputs=final_layer)
    return final_model


def freezeLayers(numOfLayers, model):
    for single_layer in model.layers[:numOfLayers]:
        single_layer.trainable = False
    for single_layer in model.layers[numOfLayers:]:
        single_layer.trainable = True

    return model


def fitModelWithTrainingData(model):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # For training data

    image_generator = datagen.flow_from_directory(TRAINING_DATA_SET_PATH,
                                                  target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                  color_mode='rgb',
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical',
                                                  shuffle=True)

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    step_size_training = image_generator.n // image_generator.batch_size  # Balance here to find optimal step_size
    model.fit_generator(generator=image_generator,
                        steps_per_epoch=step_size_training,
                        epochs=EPOCHS)
    return model


def saveModelToJson(model):
    model_json = model.to_json()
    with open("./output/model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("./output/model.h5")
    print("Saved model to disk")


def loadModelfromJson(modelPath, weightPath):
    json_file = open(modelPath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(weightPath)
    print("Loaded model from disk")
    return loaded_model


def loadSingleImage(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def trainNetwork():
    preTrainedModel = getPreTrainedModel()
    trimmedModel = freezeLayers(20, preTrainedModel)
    finalModel = fitModelWithTrainingData(trimmedModel)
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
    # finalModel = trainNetwork()
    finalModel = loadModelfromJson('./output/model.json', './output/model.h5')
    image = loadSingleImage('./data/asl-alphabet/asl_alphabet_train/asl_alphabet_train/B/B1.jpg')
    predictions = finalModel.predict(image)
    # Predictions contains a 2D array where the index of the
    # images sent in (so for a single image its only a
    # prediction[0]) and in that array is the prediction score for
    # each class A,B,C
    print(predictions)
    print(predictions.astype('int'))
    #showImageCV(predictions)


mainPipeline()
