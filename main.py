def mainPipeline():
    #finalModel = trainNetwork()
    finalModel = loadModelfromJson('/content/output/model.json', '/content/output/model-weights-best.hdf5')

    #letter = 'C'
    #imagePath = '/content/kex-dataset/asl-alphabet/asl_alphabet_test/' + letter + '/' + letter + '575.jpg'
    imagePath = '/content/kex-dataset/test-images/G/G.jpg'


    filenames = test_generator.filenames
    letter = filenames[0][0]  #Take first file and first char in filename string
    predictions = finalModel.predict_generator(test_generator,steps = len(filenames))
    #image = loadSingleImage(imagePath)
    #predictions = finalModel.predict(image)

    print('*****************************************************')
    #print(predictions)
    #loss, accuracy = evaluateModel(finalModel)
    #print("Loss: ", loss, "Accuracy: ", accuracy * 100, '%')
    print('Input was:', letter)
    top_three_preds, all_preds = getTopPredictions(predictions[0])
    print(top_three_preds)
    # print(predictions)
    #files.download('/content/output/model-weights-best.hdf5')
    #files.download('/content/output/model.json')

    print('*****************************************************')
    # print(finalModel.summary())
    # print(evaluateModel(finalModel))

    showImageCV(predictions, imagePath, letter)

    # Output updated training data structure in text-file
    #os.system("tree --filelimit=20 > project-file-structure.txt")