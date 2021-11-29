package com.knn;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class DNN {
    private static final int EPOCHS = 10;
    private static final float LEARNING_RATE = 0.001f;
    private static final Random R = new Random(0);
    private static final int RAND_MX = 10;


    private static void p(String msg) {
        System.out.println(msg);
    }

    private static float GEN_RAND() {
        return R.nextFloat() % RAND_MX;
    }

    public static void main(String[] args) {
        float[] dataSet = new float[270000];
        int numTotalDataPoints = 0;
        int numData = 0;
        try (BufferedReader br = new BufferedReader(new FileReader("csi_pruned.csv"))) {
            String line = "";
            while ((line = br.readLine()) != null) {
                String[] cells = line.split(",");
                for (String c : cells)
                    try {
                        dataSet[numTotalDataPoints++] = Float.parseFloat(c);
                    } catch (NumberFormatException nfe) {
                        nfe.printStackTrace();
                    }
                numData++;
            }
        } catch (FileNotFoundException e) {
            //Some error logging
        } catch (IOException e) {
            e.printStackTrace();
        }
        int[] originalClassArray = new int[numData];
        p(String.format("Read %d float items into the dataSet 1-D array\nTotal Data Item: %d\n", numTotalDataPoints, numData));


        // DNN : DS init
        int numTrainData = (int) (numData * 0.7f);   // 70% to be the training data
        int numTestData = numData - numTrainData;   // the rest (~30%) to be the test data

        // construct trainArray
        int numTrainPoints = numTrainData * (30 * 3 * 2);
        float[] trainArray = new float[numTrainPoints];
        int classCount = 0;
        boolean b = false;
        int p = 0;
        for (p = 0; p < (numTrainPoints + classCount); p++)
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
        float[] testArray = new float[numTestPoints];
        int classCountTest = 0;
        b = false;
        for (; p < numTotalDataPoints; p++)
            if (b && ((p - classCount) % 180 == 0)) {
                originalClassArray[classCount++] = (int) dataSet[p];
                classCountTest++;
                b = false;
            } else {
                try {
                    testArray[p - numTrainPoints - classCount] = dataSet[p];
                } catch (Exception e) {
                    e.printStackTrace();
                }
                b = true;
            }

        p(String.format("numTrainData=%d, numTrainPoints=%d, numTestData=%d, numTestPoints=%d, numTotalDataPoints=%d, classCount=%d\n",
                numTrainData, numTrainPoints, numTestData, numTestPoints, numTotalDataPoints, classCount));

        // init nnWeightArray = [feature-count = 180]
        float[] nnWeightArray = new float[180];
        for (int i = 0; i < 180; i++)
            nnWeightArray[i] = GEN_RAND();

        /*Testing matrix multiplication:* /
        auto *ta = new float[21] {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // 7x3 matrix
        // auto *tb = new float[21] {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // 3x7 matrix
        auto *tb = new float[3] {1, 2, 3}; // 3x1 matrix
        auto *tc = new float[7]; // 7x1 multiplication-resultant matrix
        matMultiplyOneDimArr(ta, tb, tc, 7, 3, 3, 1);*/

        // DNN algo. starts from here:
        // 1. Train:: -> Outcome: optimized nnWeightArray
        float[] predictedArray = new float[numTrainData]; // r1=numTrainData size
        for (int e = 0; e < EPOCHS; e++) {
            // # Forward pass train:
            predictedArray = matMultiplyOneDimArr(trainArray, nnWeightArray, numTrainData, 180, 180, 1);
            // predictedArray -> any float prediction of the probable classes
            float lossTrain = calcForwardLoss(predictedArray, numTrainData, originalClassArray);
            p(String.format("#%d: Loss = %f\n", e, lossTrain));

            float[] deltaBack = calcBackwardLoss(predictedArray, numTrainData, originalClassArray);
            // float[] deltaTrainArray = new float[numTrainPoints]; // 32400 = 180 * 180
            float[] deltaWeightArray = backwardMultiply(deltaBack, trainArray, nnWeightArray, numTrainData);

            // # Apply optimizer
            nnWeightArray = updateWeightArray(nnWeightArray, deltaWeightArray, 180, LEARNING_RATE);
        }

        predictedArray = matMultiplyOneDimArr(testArray, nnWeightArray, numTestData, 180, 180, 1);
        int numRightGuess = 0;
        for (int t = 0; t < numTestData; t++) {
            float d = predictedArray[t] - originalClassArray[numTrainData + t];
            if (d < 0) d *= (-1);
            if (d < 0.001)
                numRightGuess++;
        }
        p(String.format("Test Accuracy: %d / %d = %2.2f%%\n", numRightGuess, numTestData, ((numRightGuess * 100.0) / numTestData)));

    }

    private static float calcForwardLoss(float[] currOutArr, int currArrSize,
                                         int[] originalOutArr) {
        float sqLossTotal = 0;
        for (int i = 0; i < currArrSize; i++) {
            float d = currOutArr[i] - ((float) originalOutArr[i]);
            sqLossTotal += (d * d);
        }
        return (float) Math.sqrt(sqLossTotal / currArrSize);    // returning RMSE
    }

    private static float[] matMultiplyOneDimArr(float[] a, float[] b, int r1, int c1, int r2, int c2) {
        float[] answer = new float[r1 * c2];
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
        return answer;
    }

    // [1,2,3,4,5,6] -> [1,4,2,5,3,6], 3, 2
    private static float[] matTranspose(float[] a, int rowA, int colA) {
        // Transpose: a[ 883 x [180] ] -> m[ 180 * [883] ]
        float[] m = new float[rowA * colA];
        for (int c = 0; c < rowA; ++c)
            for (int r = 0; r < colA; ++r)
                m[r * rowA + c] = a[c * colA + r];
        return m;
    }

    private static float[] calcBackwardLoss(float[] currOutArr, int currArrSize,
                                            int[] originalOutArr) {
        float[] deltaLoss = new float[currArrSize];
        for (int i = 0; i < currArrSize; i++) {
            deltaLoss[i] = (currOutArr[i] - ((float) originalOutArr[i])) / currArrSize;
        }
        return deltaLoss;
    }

    private static float[] backwardMultiply(float[] deltaBackArray, float[] trainArray, float[] nnWeightArray, int numTrainData) {
        //deltaTrainArray += np.matmul(deltaBackArray, np.matrix.transpose(nnWeightArray))
        float[] deltaTrainArray = matMultiplyOneDimArr(deltaBackArray, nnWeightArray, numTrainData, 1, 1, 180);

        //deltaWeightArray += np.matmul(np.matrix.kernel_matTranspose(trainArray), deltaBackArray)
        float[] mt = matTranspose(trainArray, numTrainData, 180);
        return matMultiply(mt, deltaBackArray, 180, numTrainData, numTrainData, 1);
    }

    private static float[] updateWeightArray(float[] w, float[] dw, int size, float lr) {
        float[] nw = new float[size];
        for (int i = 0; i < size; i++)
            nw[i] = w[i] - (dw[i] * lr);
        return nw;
    }
}
