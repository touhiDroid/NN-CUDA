#pragma clang diagnostic push
#pragma clang diagnostic pop
#pragma ide diagnostic ignored "cppcoreguidelines-narrowing-conversions"
#pragma ide diagnostic ignored "cert-msc50-cpp"
#pragma ide diagnostic ignored "modernize-use-auto"
//
// Created by Md Touhiduzzaman on 10/24/21.
//
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<omp.h>
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

#define EPOCHS 10000 // for debug: 10, for real: 10000+
#define LEARNING_RATE 0.0001    // e=500, lr=0.0001, Test Accuracy: 294 / 379 = 77.57%, loss=0.456487->0.438812

#define TILE_DIM 32 // 9  // since 180 is a fixed known dim.
// #define BLOCK_ROWS 8 // 3 // For mat-transpose

using namespace std;

// 270000 = 1500(data-instances) * [ 181 = 1(class) + 30(sub-carriers) * [ 3(transmitter-receiver pair) * [2(r & i)]]]
float dataSet[2285000];

// REDUCE : CUDA
__global__ void kernel_calcForwardLoss(float *result,
                                       const float *currOutArr, const int *originalOutArr, int currArrSize) {
    __shared__ float sharedMemory[256];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    sharedMemory[threadIdx.x] = (tid < currArrSize) ?
                                ((currOutArr[tid] - ((float) originalOutArr[tid]))
                                 * (currOutArr[tid] - ((float) originalOutArr[tid]))) : 0;
    __syncthreads();

    // do reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + s];

        __syncthreads();
    }

    // write result for this block to global memory
    if (threadIdx.x == 0)
        atomicAdd( result, sqrt( (float) (sharedMemory[0] / currArrSize) ) );
}

// TODO TILED : CUDA
__global__ void kernel_matMultiplyTiled(float *A, float *B, float *C, int rowA, int colA, int rowB, int colB) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile_A[TILE_DIM][TILE_DIM];
    __shared__ float tile_B[TILE_DIM][TILE_DIM];

    int tiles_iterations = (rowA + TILE_DIM - 1) / TILE_DIM;

    if (row < rowA && column < colB) {

        float sum = 0;

        for (int tile = 0; tile < tiles_iterations; tile++) {
            tile_A[threadIdx.y][threadIdx.x] = A[colA * row + tile * TILE_DIM + threadIdx.x];
            tile_B[threadIdx.y][threadIdx.x] = B[tile * TILE_DIM * colB + threadIdx.y * colB + column];
            __syncthreads();

            for (int k = 0; k < TILE_DIM; k++) {
                sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
            }
            __syncthreads();
        }

        C[row * colB + column] = sum;
    }
}

__global__ void kernel_matMultiply(float *A, float *B, float *C, int rowA, int colA, int rowB, int colB) {
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (row < rowA && col < colB) {
        int sum = 0;
        for (int k = 0; k < rowB; k++)
            sum += A[row * colA + k] * B[k * colB + col];
        C[row * rowA + col] = sum;
    }
}

void matMultiply(const float *a, const float *b, float *answer, int r1, int c1, int r2, int c2) {
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

__global__ void kernel_matTranspose(float *oData, float *iData, int height, int width) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int xi = blockIdx.x * TILE_DIM + threadIdx.x;
    int yi = blockIdx.y * TILE_DIM + threadIdx.y;
    if ((xi < width) && (yi < height))
        tile[threadIdx.y][threadIdx.x] = iData[yi * width + xi];

    __syncthreads();

    xi = blockIdx.y * TILE_DIM + threadIdx.x;
    yi = blockIdx.x * TILE_DIM + threadIdx.y;
    if ((xi < height) && (yi < width))
        oData[yi * height + xi] = tile[threadIdx.x][threadIdx.y];
}

// Parallelize : CUDA
__global__ void kernel_calcBackwardLoss(float *deltaLoss, const float *currOutArr, int currArrSize,
                                        const int *originalOutArr) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < currArrSize)
        deltaLoss[tid] = (currOutArr[tid] - ((float) originalOutArr[tid])) / currArrSize;
}

// Parallelize : CUDA
__global__ void kernel_updateWeightArray(float *w, const float *dw, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size)
        w[tid] = w[tid] - (dw[tid] * LEARNING_RATE);
}

// Parallelize Weight Initialization by CUDA
/*__global__ void kernel_initWeightArray(float *w, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size)
        w[tid] = GEN_RAND; // NOLINT(bugprone-integer-division)
}*/

int main(__attribute__((unused)) int argc, __attribute__((unused)) const char *argv[]) {
    struct timespec begin, start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &begin);

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
    printf("Read %d float items into the dataSet 1-D array\nTotal Data Item: %d\n", numTotalDataPoints, numData);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    uint64_t dTimeMs = (1000000000L * (start.tv_sec - begin.tv_sec) + start.tv_nsec - begin.tv_nsec) / 1e6;
    printf("Data reading time required: %llu ms\n", (long long unsigned int) dTimeMs);


    // 1. cudaEventCreate
    float ms = 0.0;
    cudaEvent_t startCuda, stopCuda;
    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);

    // DNN : DS init
    int numTrainData = (int) (numData * 0.7f);   // 70% to be the training data
    int numTestData = numData - numTrainData;   // the rest (~30%) to be the test data

    // 2. init - cudaMallocHost & cudaMalloc
    int *originalClassArray, *d_originalClassArray;
    cudaMallocHost(&originalClassArray, numData * sizeof(int));
    cudaMalloc(&d_originalClassArray, numData * sizeof(int));

    // construct trainArray
    int numTrainPoints = numTrainData * (30 * 3 * 2);
    float *trainArray, *d_trainArray;
    size_t trainSize = numTrainPoints * sizeof(float);
    cudaMallocHost(&trainArray, trainSize);
    cudaMalloc(&d_trainArray, trainSize);
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
    float *testArray, *d_testArray;
    size_t testSize = numTestPoints * sizeof(float);
    cudaMallocHost(&testArray, testSize);
    cudaMalloc(&d_testArray, testSize);
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
    // 3. cudaMemcpy
    cudaMemcpy(d_originalClassArray, originalClassArray, numData * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trainArray, trainArray, trainSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_testArray, testArray, testSize, cudaMemcpyHostToDevice);

    printf("numTrainData=%d, numTrainPoints=%d, numTestData=%d, numTestPoints=%d, numTotalDataPoints=%d, classCount=%d\n",
           numTrainData, numTrainPoints, numTestData, numTestPoints, numTotalDataPoints, classCount);

    // init nnWeightArray = [feature-count = 180]
    float *nnWeightArray, *d_nnWeightArray;
    size_t weightArrSize = 180 * sizeof(float);
    cudaMallocHost(&nnWeightArray, weightArrSize);
    cudaMalloc(&d_nnWeightArray, weightArrSize);
    // Parallelize via CUDA -> FAiled due to rand() not being available as a device function
    // kernel_initWeightArray <<< blocksPerWtGrid, threadsPerBlock >>> (d_nnWeightArray, 180);
    // cudaMemcpy(nnWeightArray, d_nnWeightArray, weightArrSize, cudaMemcpyDeviceToHost);
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < 180; i++)
        nnWeightArray[i] = GEN_RAND; // NOLINT(bugprone-integer-division)
    cudaMemcpy(d_nnWeightArray, nnWeightArray, weightArrSize, cudaMemcpyHostToDevice);

    /*Testing matrix multiplication:* /
    float *ta = new float[21] {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // 7x3 matrix
    // float *tb = new float[21] {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // 3x7 matrix
    float *tb = new float[3] {1, 2, 3}; // 3x1 matrix
    float *tc = new float[7]; // 7x1 multiplication-resultant matrix
    matMultiply(ta, tb, tc, 7, 3, 3, 1);*/

    // DNN algo. starts from here:
    // A. Train:: -> Outcome: optimized nnWeightArray
    float *predictedArray, *d_predictedArray; // = new float[numTrainData]; // r1=numTrainData size
    size_t trainPredictArrSize = numTrainData * sizeof(float);
    cudaMallocHost(&predictedArray, trainPredictArrSize);
    cudaMalloc(&d_predictedArray, trainPredictArrSize);

    dim3 matTrDimGrid(180 / TILE_DIM, numTrainData / TILE_DIM, 1);
    dim3 matTrDimBlock(TILE_DIM, TILE_DIM, 1);
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (numTrainData + threadsPerBlock - 1) / threadsPerBlock;
    const int blocksPerWtGrid = (180 + threadsPerBlock - 1) / threadsPerBlock;

    for (int e = 0; e < EPOCHS; e++) {
        // #1. Forward pass train & calc. forward loss (by RMSE) to only understand the current status:
        /*int gridDimMult = (180 + TILE_DIM - 1 ) / TILE_DIM; // 180 = col. count of train-array
        dim3 blockSize (TILE_DIM, TILE_DIM);
        dim3 gridSize (gridDimMult, gridDimMult);
        kernel_matMultiply<<< gridSize, blockSize >>>(
                d_trainArray, d_nnWeightArray, d_predictedArray, numTrainData, 180, 180, 1);
        // predictedArray -> any float prediction of the probable classes
        cudaMemcpy(predictedArray, d_predictedArray, trainPredictArrSize, cudaMemcpyDeviceToHost);*/
        matMultiply(trainArray, nnWeightArray, predictedArray, numTrainData, 180, 180, 1);
        cudaMemcpy(d_predictedArray, predictedArray, trainPredictArrSize, cudaMemcpyHostToDevice);


        // float lossTrain = calcForwardLoss(predictedArray, numTrainData, originalClassArray);
        float *lossTrain, *d_lossTrain;
        cudaMallocHost(&lossTrain, sizeof(float));
        cudaMalloc(&d_lossTrain, sizeof(float));
        kernel_calcForwardLoss<<< blocksPerGrid, threadsPerBlock >>>(d_lossTrain, d_predictedArray,
                                                                     d_originalClassArray, numTrainData);
        cudaMemcpy(lossTrain, d_lossTrain, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("#%d: Loss = %f\n", e, *lossTrain);


        // #2. Backward propagation:
        //  a. Calculate relative deviation of the predictions to have deltaBack
        //          -> (still in confusion: why deltaTrainArray may be needed?)
        //  b. Use the deltaBack array to calculate the deltaWeightArray
        // 2.a
        float *deltaBack, *d_deltaBack;//  = new float[numTrainData];
        cudaMallocHost(&deltaBack, numTrainData * sizeof(float));
        cudaMalloc(&d_deltaBack, numTrainData * sizeof(float));
        kernel_calcBackwardLoss<<< blocksPerGrid, threadsPerBlock >>>(d_deltaBack, d_predictedArray,
                                                                      numTrainData, d_originalClassArray);
        cudaMemcpy(deltaBack, d_deltaBack, numTrainData * sizeof(float), cudaMemcpyDeviceToHost);
        float *deltaTrainArray = new float[numTrainPoints]; // 32400 = 180 * 180
        matMultiply(deltaBack, nnWeightArray, deltaTrainArray, numTrainData, 1, 1, 180);

        // 2.b
        float *deltaWeightArray, *d_deltaWeightArray;    // same as the weight-array
        cudaMallocHost(&deltaWeightArray, weightArrSize);
        cudaMalloc(&d_deltaWeightArray, weightArrSize);
        float *mt, *d_mt;
        cudaMallocHost(&mt, trainSize);
        cudaMalloc(&d_mt, trainSize);
        kernel_matTranspose<<< matTrDimGrid, matTrDimBlock >>>(d_mt, d_trainArray, numTrainData, 180);
        cudaMemcpy(mt, d_mt, trainSize, cudaMemcpyDeviceToHost);
        matMultiply(mt, deltaBack, deltaWeightArray, 180, numTrainData, numTrainData, 1);


        // #3. Update the nnWeightArray by applying y=mx+c derivative,
        // where y->new weights, x->old weights, m=learning_rate, c=bias (0 in this implementation)
        cudaMemcpy(d_nnWeightArray, nnWeightArray, weightArrSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_deltaWeightArray, deltaWeightArray, weightArrSize, cudaMemcpyHostToDevice);
        kernel_updateWeightArray <<< blocksPerWtGrid, threadsPerBlock >>> (d_nnWeightArray, d_deltaWeightArray, 180);
        cudaMemcpy(nnWeightArray, d_nnWeightArray, weightArrSize, cudaMemcpyDeviceToHost);
        if ((*lossTrain) < 1e-8)
            break;
    }

    // B. Test:: -> Outcome: optimized nnWeightArray
    predictedArray = new float[numTestData];
    matMultiply(testArray, nnWeightArray, predictedArray, numTestData, 180, 180, 1);
    int numRightGuess = 0;
    for (int t = 0; t < numTestData; t++) {
        if (predictedArray[t] < 0.0f) predictedArray[t] *= (-1);
        int c = predictedArray[t] < 0.5 ? 0 : 1;
        if (c == originalClassArray[numTrainData + t])
            numRightGuess++;
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t dAlgoTimeMsg = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    printf("Test Accuracy: %d / %d = %2.2f%%\nRuntime: %llu ms\n", numRightGuess, numTestData,
           ((numRightGuess * 100.0) / numTestData), (long long unsigned int) dAlgoTimeMsg);


    // 6. free mem. & destroy events
    cudaFreeHost(trainArray);
    cudaFree(d_trainArray);

    cudaFreeHost(testArray);
    cudaFree(d_testArray);

    cudaFreeHost(nnWeightArray);
    cudaFree(d_nnWeightArray);

    cudaFreeHost(predictedArray);
    cudaFree(d_predictedArray);

    cudaFreeHost(originalClassArray);
    cudaFree(d_originalClassArray);

    cudaEventDestroy(startCuda);
    cudaEventDestroy(stopCuda);

    return 0;
}
