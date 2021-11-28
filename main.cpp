//
// Created by Md Touhiduzzaman on 11/24/21.
//
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "cmath"

#define RAND_MX 10
#define GEN_RAND ( ((float)rand())/((float)RAND_MX) )

#define EPOCHS 10 // for debug: 10, for real: 10000+

using namespace std;

// 270000 = 1500(data-instances) * [ 181 = 1(class) + 30(sub-carriers) * [ 3(transmitter-receiver pair) * [2(r & i)]]]
float dataSet[270000];


void matMultiplyOneDimArr(float *a, float *b, float *answer, int r1, int c1, int r2, int c2) {
    float cellSum = 0;
    int i, j, k;
    // r1==c2, r2==c1
    for (i = 0; i < r1; i++) {
        for (j = 0; j < c2; j++) {
            for (k = 0; k < r2; k++)
                cellSum += a[i * c1 + k] * b[k * c2 + j];
            answer[i * c2 + j] = cellSum;
            cellSum = 0;
        }
    }
}

int main(int argc, const char *argv[]) {
    ifstream csiPrunedCsvFile("csi_pruned.csv");
    string csvStr;
    int numTotalDataPoints = 0;
    int numData = 0;
    while (getline(csiPrunedCsvFile, csvStr)) {
        stringstream csvStrStream(csvStr);
        string singleFloatItem;
        while (getline(csvStrStream, singleFloatItem, ','))
            dataSet[numTotalDataPoints++] = atof(singleFloatItem.c_str());
        numData++;
    }
    int *originalClassArray = new int[numData];
    printf("Read %d float items into the dataSet 1-D array\nTotal Data Item: %d\n", numTotalDataPoints, numData);


    // DNN : DS init
    int numTrainData = (int) (numData * 0.7f);   // 70% to be the training data
    int numTestData = numData - numTrainData;   // the rest (~30%) to be the test data

    // construct trainArray
    int numTrainPoints = numTrainData * (30 * 3 * 2);
    float *trainArray = new float[numTrainPoints];
    int classCount = 0;
    for (int i = 0; i < numTrainPoints; i++)
        if (i % 181 == 0)
            originalClassArray[classCount++] = dataSet[i];
        else
            trainArray[i-classCount] = dataSet[i];

    // DONE : ensure that `numTotalDataPoints = numTrainPoints + numTestPoints + classCount`

    // construct testArray
    int numTestPoints = numTestData * (30 * 3 * 2);
    float *testArray = new float[numTestPoints];
    for (int i = numTrainPoints + 1; i < numTotalDataPoints; i++)
        if (i!=0 && i % 181 == 0)
            originalClassArray[classCount++] = dataSet[i];
        else
            testArray[i - numTrainPoints - 1 - classCount] = dataSet[i];

    printf("numTrainData=%d, numTrainPoints=%d, numTestData=%d, numTestPoints=%d, numTotalDataPoints=%d, classCount=%d",
           numTrainData, numTrainPoints, numTestData, numTestPoints, numTotalDataPoints, classCount);

    // init nnWeightArray = [181][numTrainData] (total size = numTrainPoints)
    auto *nnWeightArray = new float[numTrainPoints];
    for (int i = 0; i < numTrainPoints; i++)
        nnWeightArray[i] = GEN_RAND;

    /*Testing matrix multiplication:
    auto *ta = new float[21] {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // 7x3 matrix
    auto *tb = new float[21] {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // 3x7 matrix
    auto *tc = new float[49]; // 7x7 multiplication-resultant matrix
    matMultiplyOneDimArr(ta, tb, tc, 7, 3, 3, 7);*/

    // DNN algo. starts from here:
    auto *multResArray = new float[numTrainData * numTrainData]; // r1*c2 size
    for (int e = 0; e < EPOCHS; e++) {
        /*#Forward pass train:
            nnOutput = forwardMult(inputArray,nnWeights)*/
        matMultiplyOneDimArr(trainArray, nnWeightArray, multResArray, numTrainData, 180, 180, numTrainData);
        /*lossTrain = forwardLoss(nnOutput,outputArray)
        historyTrain.append(lossTrain)

    #Forward pass test:
        nnTest = forwardMult(inputTest,nnWeights)
        lossTest = forwardLoss(nnTest,outputTest)
        historyTest.append(lossTest)
        #Print Loss every 50 epochs:
        if(i%10==0):
            print("Epoch: " + str(i) + " Loss (train): " + "{0:.3f}".format(lossTrain) + " Loss (test): " + "{0:.3f}".format(lossTest))

    #Backpropagate
        deltaOutput = backwardLoss(nnOutput,outputArray)
        backwardMult(deltaOutput,inputArray,nnWeights,deltaInput,deltaWeights)

    #Apply optimizer
        updateWeights(nnWeights,deltaWeights, learningRate)

    #Reset deltas
        deltaInput = np.zeros((batchSize,inputSize))
        deltaWeights = np.zeros((inputSize,outputSize))
        deltaOutput = np.zeros((inputSize,outputSize))

    #Start new epoch
        i = i+1*/
    }

    return 0;
}
