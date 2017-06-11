#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "LayerTop.hpp"

#define LayerNum 4

using namespace std;

int main(int argc, char **argv) {
    int weightSize[LayerNum];
    int biasSize[LayerNum];
    int inputFmSize[LayerNum];
    int resultSize;
    int weightOffset, biasOffset, inputFmOffset[LayerNum];
    int resultOffset;

    //parse arguments
    if(argc != 3) cout << "error input argc" << endl;
    int iterNum = atoi(argv[1]);
    int batchSize = atoi(argv[2]);

    weightSize[0] = 20 * 1 * 5 * 5;
    weightSize[1] = 50 * 20 * 5 * 5;
    weightSize[2] = 500 * 800;
    weightSize[3] = 10 * 500;

    biasSize[0] = 20;
    biasSize[1] = 50;
    biasSize[2] = 500;
    biasSize[3] = 10;

    inputFmSize[0] = batchSize * 1 * 28 * 28;
    inputFmSize[1] = batchSize * 20 * 12 * 12;
    inputFmSize[2] = batchSize * 800;
    inputFmSize[3] = batchSize * 500;

    resultSize = batchSize * 10;


    weightOffset = 0;
    biasOffset = 0;


    int weightSum = 0;
    int biasSum = 0;
    for(int i = 0; i < LayerNum; ++i) {
        weightSum += weightSize[i];
        biasSum += biasSize[i];
    }

    biasOffset = weightSum;
    inputFmOffset[0] = weightSum + biasSum;
    inputFmOffset[1] = inputFmOffset[0] + inputFmSize[0];
    inputFmOffset[2] = inputFmOffset[1] + inputFmSize[1];
    inputFmOffset[3] = inputFmOffset[2] + inputFmSize[2];
    resultOffset = inputFmOffset[3] + inputFmSize[3];

    float *dram = new float[resultOffset + resultSize];
    cout << "Total Length: " << (resultOffset + resultSize) << endl;

    FILE *fhd = fopen("/home/hht/work/my_cnn/weight.bin", "rb");
    int readLen = fread(dram+weightOffset, sizeof(float), weightSum, fhd);
    //for(int i = 0; i < weightSum; ++i) cout << *(dram+weightOffset+i) << endl;
    readLen = fread(dram+biasOffset, sizeof(float), biasSum, fhd);
    //for(int i = 0; i < biasSum; ++i) cout << *(dram+biasOffset+i) << endl;
    fclose(fhd);

    //cout << weightOffset << "\t" << biasOffset << "\t" << inputFmOffset[0] << "\t" <<
    //    inputFmOffset[1] << "\t" << inputFmOffset[1] << "\t" << inputFmOffset[2] << "\t" << 
    //    inputFmOffset[3] << "\t" << endl;

    int correcNum = 0;
    int recordLen = iterNum;
    fhd = fopen("/home/hht/work/my_cnn/img.bin", "rb");
    FILE *labelfd = fopen("/home/hht/work/my_cnn/label.bin", "rb");
    if(labelfd == NULL) cout << "label.bin file not found." << endl;
    for(int k = 0; k < ceil(1.0*iterNum/batchSize); ++k) {

        int bMax;
        if(recordLen < batchSize)
            bMax = recordLen;
        else bMax = batchSize;

        recordLen -= batchSize;
            
        readLen = fread(dram+inputFmOffset[0], sizeof(float), bMax*(inputFmSize[0]/batchSize), fhd);

        char *golden = new char[bMax];
        readLen = fread(golden, sizeof(char), bMax, labelfd);
        //cout << "bMax: " << bMax << endl;
        for(int b = 0; b < bMax; ++b) {

            LayerTop(dram, dram, dram, LayerCfgType(
                        weightOffset, biasOffset, inputFmOffset[0]+b*(inputFmSize[0]/batchSize),
                        inputFmOffset[1]+b*(inputFmSize[1]/batchSize),
                        0, 0, 1, 0,
                        24, 24, 20, 1, 5, 1, 2, 2));
            
            //for(int i = 0; i < inputFmSize[1]; ++i) cout << "inputFmSize1: " << *(dram+inputFmOffset[1]+i) << endl;
            //cout << "output2: " << inputFmOffset[2]+b*(inputFmSize[2]/batchSize) << endl;
            LayerTop(dram, dram, dram, LayerCfgType(
                        weightOffset+weightSize[0], biasOffset+biasSize[0],
                        inputFmOffset[1]+b*(inputFmSize[1]/batchSize),
                        inputFmOffset[2]+b*(inputFmSize[2]/batchSize),
                        0, 0, 1, 0,
                        8, 8, 50, 20, 5, 1, 2, 2));
        }

        //for(int i = 0; i < inputFmSize[2]; ++i) cout << "inputFmSize2: " << *(dram+inputFmOffset[2]+i) << endl;
        LayerTop(dram, dram, dram, LayerCfgType(
                    inputFmOffset[2], biasOffset+biasSize[0]+biasSize[1],
                    weightOffset+weightSize[0]+weightSize[1], inputFmOffset[3],
                    0, 1, 0, 1,
                    20, 25, bMax, 800, 1, 1, 1, 1));

        //for(int i = 0; i < inputFmSize[3]; ++i) cout << "inputFmSize3: " << *(dram+inputFmOffset[3]+i) << endl;
        LayerTop(dram, dram, dram, LayerCfgType(
                    inputFmOffset[3], biasOffset+biasSize[0]+biasSize[1]+biasSize[2],
                    weightOffset+weightSize[0]+weightSize[1]+weightSize[2], resultOffset,
                    0, 0, 0, 1,
                    1, 10, bMax, 500, 1, 1, 1, 1));

        //Calc results
        int index = 0;
        float max, readData;
        int tmpSize = resultSize/batchSize;
        //cout << "tmpSize = " << tmpSize << endl;
        for(int j = 0; j < bMax; ++j) {
            for(int i = 0; i < tmpSize; ++i) {
                readData = *(dram+resultOffset+j*tmpSize+i);
                cout <<"\t" << readData;
                if(i == 0) {
                    max = readData;
                    index = 0;
                }
                if(max < readData) {
                    max = readData;
                    index = i;
                }
            }
            cout << endl;

            correcNum += (index == (int)golden[j]);
            cout << k*batchSize+j+1 <<": Calc Number is " << index << "\tExpected is " << ((int)golden[j]) << endl;
            cout << "Accuracy is " << 1.0*correcNum/(k*batchSize+j+1) << endl;
        }

    }
    fclose(fhd);
    fclose(labelfd);

    return 0;

}



