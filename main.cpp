//
// Created by Md Touhiduzzaman on 11/24/21.
//
#include "fstream"
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

float dataSet[270000]; // 270000 = 1500(data-instances) * [ 30(sub-carriers) * [ 3(transmitter-receiver pair) * [2(r & i)]]]
int main() {
    ifstream csiPrunedCsvFile("csi_pruned.csv");
    string csvStr;
    int i = 0;
    while (getline(csiPrunedCsvFile, csvStr)) {
        stringstream csvStrStream(csvStr);
        string singleFloatItem;
        while (getline(csvStrStream, singleFloatItem, ','))
            dataSet[i++] = atof(singleFloatItem.c_str());
    }
    printf("Read %d float items into the dataSet 1-D array\n", i);

    return 0;
}
