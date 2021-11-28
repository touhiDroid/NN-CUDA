import numpy as np
import matplotlib.pyplot as plt

# config
inputSize = 50
outputSize = 30
batchSize = 100
testSize = 50
epochs = 40000
learningRate = 0.01
seed = 123456


# create functions
# Operator example in forward and backward (Mult)
def forwardMult(A, B):
    return np.matmul(A, B)


def backwardMult(dC, A, B, dA, dB):
    dA += np.matmul(dC, np.matrix.transpose(B))
    dB += np.matmul(np.matrix.transpose(A), dC)


# Loss example in forward and backward (RMSE)
def forwardloss(predictedOutput, realOutput):
    if (predictedOutput.shape == realOutput.shape):
        loss = np.mean(0.5 * np.square(predictedOutput - realOutput))
    else:
        print("Shape of arrays not the same")
    return loss


def backwardloss(predictedOutput, realOutput):
    if (predictedOutput.shape == realOutput.shape):
        deltaOutput = (predictedOutput - realOutput) / predictedOutput.size
    else:
        print("Shape of arrays not the same")
    return deltaOutput


# Optimizer example (SGD)
def updateweights(W, dW, learningRate):
    W -= learningRate * dW


if __name__ == '__main__':
    # Generation of fake dataset - we generate random inputs and weights and calculate outputs
    np.random.seed(seed)
    inputArray = np.random.uniform(-5, 5, (batchSize, inputSize))
    weights = np.random.uniform(-5, 5, (inputSize, outputSize))
    outputArray = np.matmul(inputArray, weights)
    inputTest = np.random.uniform(-5, 5, (testSize, inputSize))
    outputTest = np.matmul(inputTest, weights)

    # initialization of NN by other random weights
    nnWeights = np.random.uniform(-3, 3, (inputSize, outputSize))
    deltaweights = np.zeros((inputSize, outputSize))
    deltainput = np.zeros((batchSize, inputSize))
    deltaoutput = np.zeros((inputSize, outputSize))

    historyTrain = []  # Used to record the history of loss
    historyTest = []
    i = 1
    while i <= epochs:
        # Forward pass train:
        nnOutput = forwardMult(inputArray, nnWeights)
        lossTrain = forwardloss(nnOutput, outputArray)
        historyTrain.append(lossTrain)

        # Forward pass test:
        nnTest = forwardMult(inputTest, nnWeights)
        lossTest = forwardloss(nnTest, outputTest)
        historyTest.append(lossTest)
        if lossTest <= 0.001:
            break
        # Print Loss every 50 epochs:
        if i % 50 == 0:
            print("Epoch: " + str(i) + " Loss (train): " + "{0:.3f}".format(
                lossTrain) + " Loss (test): " + "{0:.3f}".format(lossTest))

        # Backpropagate
        deltaoutput = backwardloss(nnOutput, outputArray)
        backwardMult(deltaoutput, inputArray, nnWeights, deltainput, deltaweights)

        # Apply optimizer
        updateweights(nnWeights, deltaweights, learningRate)

        # Reset deltas
        deltainput = np.zeros((batchSize, inputSize))
        deltaweights = np.zeros((inputSize, outputSize))
        deltaoutput = np.zeros((inputSize, outputSize))

        # Start new epoch
        i = i + 1
