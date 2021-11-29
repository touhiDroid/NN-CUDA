#pragma clang diagnostic push
#pragma clang diagnostic pop
#pragma ide diagnostic ignored "cppcoreguidelines-narrowing-conversions"
#pragma ide diagnostic ignored "cert-msc50-cpp"
#pragma ide diagnostic ignored "modernize-use-auto"
//
// Created by Md Touhiduzzaman on 10/24/21.
//
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
//#include <bits/stdc++.h>
//#include <omp.h>
#include <pthread.h>
#include "cmath"

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

// #define RAND_MX 1
#define GEN_RAND ( (rand() % 10) / 10 )

#define EPOCHS 500 // for debug: 10, for real: 10000+
#define LEARNING_RATE 0.0001    // e=500, lr=0.0001, Test Accuracy: 294 / 379 = 77.57%, loss=0.456487->0.438812

using namespace std;

// 270000 = 1500(data-instances) * [ 181 = 1(class) + 30(sub-carriers) * [ 3(transmitter-receiver pair) * [2(r & i)]]]
float dataSet[270000];

// TODO REDUCE : MPI / CUDA
float calcForwardLoss(const float *currOutArr, int currArrSize,
                      const int *originalOutArr, int startOffSetOriginalArr = 0) {
    float sqLossTotal = 0;
    for (int i = 0; i < currArrSize; i++) {
        float d = currOutArr[i] - ((float) originalOutArr[startOffSetOriginalArr + i]);
        sqLossTotal += (d * d);
    }
    return sqrt(sqLossTotal / currArrSize);    // returning RMSE
}

// TODO TILED : CUDA
void matMultiplyOneDimArr(const float *a, const float *b, float *answer, int r1, int c1, int r2, int c2) {
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

// TODO CUDA
// [1,2,3,4,5,6] -> [1,4,2,5,3,6], 3, 2
void matTranspose(const float *a, float *m, int rowA, int colA) {
    // Transpose: a[ 883 x [180] ] -> m[ 180 * [883] ]
    for (int c = 0; c < rowA; ++c)
        for (int r = 0; r < colA; ++r)
            m[r * rowA + c] = a[c * colA + r];
}

// OPENMP : Parallelize
void calcBackwardLoss(float *deltaLoss, const float *currOutArr, int currArrSize,
                      const int *originalOutArr, int startOffSetOriginalArr = 0) {
    for (int i = 0; i < currArrSize; i++) {
        deltaLoss[i] = (currOutArr[i] - ((float) originalOutArr[startOffSetOriginalArr + i])) / currArrSize;
    }
}

// Shall have nested CUDA calls
void backwardMultiply(float *deltaBackArray, const float *trainArray, const float *nnWeightArray,
                      float *deltaTrainArray, float *deltaWeightArray, int numTrainData) {
    //deltaTrainArray += np.matmul(deltaBackArray, np.matrix.transpose(nnWeightArray))
    matMultiplyOneDimArr(deltaBackArray, nnWeightArray, deltaTrainArray, numTrainData, 1, 1, 180);

    //deltaWeightArray += np.matmul(np.matrix.transpose(trainArray), deltaBackArray)
    float *mt = new float[numTrainData * 180];
    matTranspose(trainArray, mt, numTrainData, 180);
    matMultiplyOneDimArr(mt, deltaBackArray, deltaWeightArray, 180, numTrainData, numTrainData, 1);
}

// TODO Parallelize : MPI / OpenMP / CUDA
void updateWeightArray(float *w, const float *dw, int size) {
    for (int i = 0; i < size; i++)
        w[i] -= (dw[i] * LEARNING_RATE);
}

int main(__attribute__((unused)) int argc, __attribute__((unused)) const char *argv[]) {
    ifstream csiPrunedCsvFile("csi_pruned.csv");
    string csvStr;
    int numTotalDataPoints = 0;
    int numData = 0;
    while (getline(csiPrunedCsvFile, csvStr)) {
        stringstream csvStrStream(csvStr);
        string singleFloatItem;
        while (getline(csvStrStream, singleFloatItem, ','))
            dataSet[numTotalDataPoints++] = atof(singleFloatItem.c_str()); // NOLINT(cert-err34-c)
        numData++;
    }
    int *originalClassArray = new int[numData];
    printf("Read %d float items into the dataSet 1-D array\nTotal Data Item: %d\n", numTotalDataPoints, numData);


    // DNN : DS init
    int numTrainData = (int) (numData * 0.7f);   // 70% to be the training data
    int numTestData = numData - numTrainData;   // the rest (~30%) to be the test data

    // construct trainArray
    int numTrainPoints = numTrainData * (30 * 3 * 2);
    float *trainArray = new float[numTrainPoints]; // NOLINT(modernize-use-auto)
    int classCount = 0;
    bool b = false;
    int p = 0;
    // TODO : Potential Parallelization by MPI / OpenMP / CUDA - but wd be difficult due to inc.
    for (; p < (numTrainPoints + classCount); p++)
        if (b && ((p - classCount) % 180 == 0)) {
            originalClassArray[classCount++] = (int) dataSet[p];
            b = false;
        } else {
            trainArray[p - classCount] = dataSet[p];
            b = true;
        }
    originalClassArray[classCount++] = (int) dataSet[p++];

    // Note: `numTotalDataPoints = numTrainPoints + numTestPoints + classCount`

    // construct testArray
    int numTestPoints = numTestData * (30 * 3 * 2);
    float *testArray = new float[numTestPoints];
    int classCountTest = 0;
    b = false;
    // TODO : Potential Parallelization by MPI / OpenMP / CUDA - but wd be difficult due to inc.
    for (; p < numTotalDataPoints; p++)
        if (b && ((p - classCount) % 180 == 0)) {
            originalClassArray[classCount++] = (int) dataSet[p];
            classCountTest++;
            b = false;
        } else {
            testArray[p - numTrainPoints - classCount] = dataSet[p];
            b = true;
        }

    printf("numTrainData=%d, numTrainPoints=%d, numTestData=%d, numTestPoints=%d, numTotalDataPoints=%d, classCount=%d\n",
           numTrainData, numTrainPoints, numTestData, numTestPoints, numTotalDataPoints, classCount);

    // init nnWeightArray = [feature-count = 180]
    float *nnWeightArray = new float[180];
    // TODO Parallelize via OpenMP / MPI / CUDA
    for (int i = 0; i < 180; i++)
        nnWeightArray[i] = GEN_RAND; // NOLINT(bugprone-integer-division)

    /*Testing matrix multiplication:* /
    float *ta = new float[21] {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // 7x3 matrix
    // float *tb = new float[21] {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // 3x7 matrix
    float *tb = new float[3] {1, 2, 3}; // 3x1 matrix
    float *tc = new float[7]; // 7x1 multiplication-resultant matrix
    matMultiplyOneDimArr(ta, tb, tc, 7, 3, 3, 1);*/

    // DNN algo. starts from here:
    // 1. Train:: -> Outcome: optimized nnWeightArray
    float *predictedArray = new float[numTrainData]; // r1=numTrainData size
    for (int e = 0; e < EPOCHS; e++) {
        // # Forward pass train:
        matMultiplyOneDimArr(trainArray, nnWeightArray, predictedArray, numTrainData, 180, 180, 1);
        // predictedArray -> any float prediction of the probable classes
        float lossTrain = calcForwardLoss(predictedArray, numTrainData, originalClassArray);
        printf("#%d: Loss = %f\n", e, lossTrain);

        float *deltaBack = new float[numTrainData];
        calcBackwardLoss(deltaBack, predictedArray, numTrainData, originalClassArray);
        float *deltaTrainArray = new float[numTrainPoints]; // 32400 = 180 * 180
        float *deltaWeightArray = new float[180];    // same as the weight-array
        backwardMultiply(deltaBack, trainArray, nnWeightArray, deltaTrainArray, deltaWeightArray, numTrainData);

        // # Apply optimizer
        updateWeightArray(nnWeightArray, deltaWeightArray, 180);
        if (lossTrain < 1e-8)
            break;
    }

    predictedArray = new float[numTestData];
    matMultiplyOneDimArr(testArray, nnWeightArray, predictedArray, numTestData, 180, 180, 1);
    int numRightGuess = 0;
    for (int t = 0; t < numTestData; t++) {
        if (predictedArray[t] < 0.0f) predictedArray[t] *= (-1);
        int c = predictedArray[t] < 0.5 ? 0 : 1;
        if (c == originalClassArray[numTrainData + t])
            numRightGuess++;
    }
    printf("Test Accuracy: %d / %d = %2.2f%%\n", numRightGuess, numTestData, ((numRightGuess * 100.0) / numTestData));

    return 0;
}