import sys
import json
import os.path

sys.path.append(os.path.join(sys.path[0], '..'))
import src.NeuralNetwork as NN

def loadFileBytes(file: str) -> bytes:
    with open(file, "rb") as f:
        return f.read()

def loadMnistDataSet(labelFile: str, dataFile: str) -> dict:
    dataSets: list = []
    labelBytes: bytes = loadFileBytes(labelFile)
    dataBytes: bytes = loadFileBytes(dataFile)

    readHeadData: int = 16
    readHeadLabel: int = 8

    while readHeadData < len(dataBytes):
        dataSet: dict = {
            "label": int.from_bytes(labelBytes[readHeadLabel:readHeadLabel+1]),
            "data": []
        }
        readHeadLabel += 1

        for i in range(28):
            for k in range(28):
                dataSet['data'].append(
                    int.from_bytes(dataBytes[readHeadData:readHeadData+1]) / 255.0
                )
                readHeadData += 1

        dataSets.append(dataSet)

    return dataSets

# nn setup ------------------------------------------------

nn_storage_file: str = './tests/NeuralNetworkMnist.nn'

nn = NN.NeuralNetwork()
nn.inputSize = 1+(28*28)
nn.hiddenNumber = 0
nn.outputSize = 10
nn.outputActivationFunction = nn.ACTIVATION_FN_SIGMOID

if os.path.isfile(nn_storage_file):
    nn.weights = json.load(open(nn_storage_file))

# training -------------------------------------------------

trainingCycles: int = 1

if trainingCycles > 0:
    trainingSets: dict = loadMnistDataSet(
        "./mnist-data/train-labels.idx1-ubyte",
        "./mnist-data/train-images.idx3-ubyte"
    )
    print('training sets loaded')

    all: int = len(trainingSets)
    for i in range(trainingCycles):            
        print('starting training cycle ' + str(i+1))
        num: int = 0
        for trainingSet in trainingSets:
            num+=1
            print(str(num) + "/" + str(all), end='\r')

            input: list = trainingSet["data"].copy()
            input.append(1) # bias
            expected: list = [0,0,0,0,0,0,0,0,0,0];
            expected[trainingSet["label"]] = 1;
            nn.train(input, expected)

    json.dump(nn.weights, open(nn_storage_file, 'w'))

# testing ------------------------------------------------

testingCycles: int = 1

if testingCycles > 0:
    testingSets: dict = loadMnistDataSet(
        "./mnist-data/t10k-labels.idx1-ubyte",
        "./mnist-data/t10k-images.idx3-ubyte"
    )
    print('testing sets loaded')

    all: int = len(testingSets)
    for i in range(testingCycles):
        print('starting testingSets cycle ' + str(i+1))

        correct: int = 0
        incorrect: int = 0
        num: int = 0
        for testingSet in testingSets:
            num+=1
            print(str(num) + "/" + str(all), end='\r')

            input: list = testingSet["data"].copy()
            input.append(1) # bias
            output: list = nn.execute(input)
            
            if output.index(max(output)) == testingSet["label"]: correct += 1
            else: incorrect += 1

        print('ended testingSets cycle ' + str(i+1))
        print('correct:', correct)
        print('incorrect:', incorrect)
