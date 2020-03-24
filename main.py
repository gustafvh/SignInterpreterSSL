from PIL import Image
import numpy as np
# Weights are initiliased at radom so this makes our model deterministic (same result each time)
np.random.seed(123) # for reproducibility
import pandas as pd
import json
from skimage import transform
from matplotlib import pyplot as plt

from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint


from keras.utils import plot_model


import cv2
import os


#### Variables ######

TRAINING_DATA_SET_PATH = './data/asl-alphabet-medium/asl_alphabet_train'
VALIDATION_DATA_SET_PATH = './data/asl-alphabet-medium/asl_alphabet_test'
EPOCHS = 5 #5 gives 54% accuracy
BATCH_SIZE = 32
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
DATASET_CATEGORIES = 29


####################

def getPreTrainedModel():
    pre_trained_base_model = MobileNet(weights='imagenet',
                                       include_top=False)  # Remove last neuron layer (that contains 1000 neurons)

    my_model = pre_trained_base_model.output
    my_model = GlobalAveragePooling2D()(my_model)
    my_model = Dense(1000, activation='relu')(
        my_model)  # Add layer. Layer #1: 1000 nodes, and rectified linear activation function
    my_model = Dense(1000, activation='relu')(my_model)  # Layer #2
    my_model = Dense(500, activation='relu')(my_model)
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
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # For training data

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(TRAINING_DATA_SET_PATH,
                                                        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                        color_mode='rgb',
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical',
                                                        shuffle=True)

    validation_generator = val_datagen.flow_from_directory(
        VALIDATION_DATA_SET_PATH,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Checkpoints: save only the best performing model weights as file
    checkpoint = ModelCheckpoint('./output/model-weights.best.hdf5',
                                 monitor='val_accuracy', verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False, mode='max')
    callbacks_list = [checkpoint]

    step_size_training = train_generator.n // train_generator.batch_size  # Balance here to find optimal step_size
    history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_training,
                        epochs=EPOCHS,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.n // BATCH_SIZE,
                        callbacks=callbacks_list)

    # # Save training history as file. For ex. loss and accuracy in each epoch
    # with open('./output/history.json', 'w') as file:
    #     json.dump(history.history, file)

    return model


def trainNetwork():
    preTrainedModel = getPreTrainedModel()
    trimmedModel = freezeLayers(20, preTrainedModel)
    finalModel = fitModel(trimmedModel)
    saveModelToJson(finalModel)
    return finalModel


def plotEpochPerformance(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # This will plot a graph of the model and save it to a file:
    #plot_model(model, to_file='model.png')

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


def showImageCV(preds, imagePath, letter):
    # find the class label index with the largest corresponding
    # probability
    i = preds.argmax(axis=1)[0]
    # label = lb.classes_[i]
    # draw the class label + probability on the output image
    text = "{}: {:.2f}%".format(letter, preds[0][i] * 100)

    image = cv2.imread(imagePath)
    output = image.copy()
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)
    # show the output image
    cv2.imshow("Image", output)
    cv2.waitKey(0)


def getTopPredictions(preds):
    #top_preds = preds.argmax(axis=1)[0]

    predsDict = {
        'A': 0, 'B': 0, 'C': 0, 'D': 0, 'del': 0, 'E': 0, 'F': 0, 'G': 0,
        'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'nothing': 0,
        'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'space': 0, 'T': 0,
        'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0
    }
    #map preds index with probability to correct letter from dictonary
    for i, key in enumerate(predsDict, start=0):
        predsDict[key] = preds[i]


    # sort by dictonary value and returns as list
    all_preds = sorted(predsDict.items(), reverse=True, key=lambda x: x[1])

    top_preds = [all_preds[0], all_preds[1], all_preds[2]]
    # for i, key in enumerate(all_preds, start=0):
    #     top_preds.append(

    #print(all_preds.keys()[0])
    # Convert from float into percentages
    #top_preds = pd.DataFrame(preds / float(np.sum(preds))).applymap(lambda preds: '{:.2%}'.format(preds)).values
    #print(preds[0])

    return top_preds, all_preds


def evaluateModel(model):
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    validation_generator = val_datagen.flow_from_directory(
        VALIDATION_DATA_SET_PATH,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    score = model.evaluate_generator(validation_generator,
                                     validation_generator.n / BATCH_SIZE,
                                     workers=12, use_multiprocessing=True,
                                     verbose=1)
    loss, accuracy = score[0], score[1]
    return loss, accuracy


def mainPipeline():
    #finalModel = trainNetwork()
    finalModel = loadModelfromJson('./output/model.json', './output/model.h5')


    letter = 'A'
    imagePath = './data/asl-alphabet/asl_alphabet_train/' + letter + '/' + letter + '575.jpg'
    #imagePath = './data/test-images/' + letter + '/' + letter + '.jpg'
    image = loadSingleImage(imagePath)
    predictions = finalModel.predict(image,
                                     workers=12,
                                     use_multiprocessing=True)


    print('*****************************************************')
    #print(predictions)
    loss, accuracy = evaluateModel(finalModel)
    print("Loss: ", loss, "Accuracy: ", accuracy*100, '%')
    print('Input was:', letter)
    top_three_preds, all_preds = getTopPredictions(predictions[0])
    print(top_three_preds)
    # print(predictions)


    print('*****************************************************')
    #print(finalModel.summary())
    #print(evaluateModel(finalModel))

    #showImageCV(predictions, imagePath, letter)

    #Output updated training data structure in text-file
    os.system("tree --filelimit=20 > project-file-structure.txt")




mainPipeline()



