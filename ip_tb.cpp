#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "ip.hpp"
#include "caffe.pb.h"

#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

const int kProtoReadBytesLimit = INT_MAX;

using namespace std;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  if(fd == -1)
    cout << "File not found: " << filename << endl;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}


int main(int argc, char** argv) {
    int isPadding, isRelu, isPoolingMax;
    int cfgRow, cfgCol, cfgM, cfgN, cfgK, cfgS;
    int cfgPoolK, cfgPoolS;
    int weightOffset, biasOffset, inFmOffset, outFmOffset;
    int weightSize, biasSize, inputFmSize, outputFmSize;
    float *dram;
    FILE *labelfd, *inputfd, *fd;
    int correcNum = 0;

    caffe::NetParameter net;
    const char weightsName[] = "lenet_iter_10000.caffemodel";
    bool success = ReadProtoFromBinaryFile(weightsName, &net);
    cout << "Net Name: " << net.name() << endl;
    cout << "Layers Num: " << net.layer_size() << endl;

    inputfd = fopen("img.bin", "rb");
    if(inputfd == NULL) cout << "img.bin file not found." << endl;
    labelfd = fopen("label.bin", "rb");
    if(labelfd == NULL) cout << "label.bin file not found." << endl;

    if(argc != 3) cout << "error input argc" << endl;
    int iterNum = atoi(argv[1]);
    int batch = atoi(argv[2]);

    for(int k = 0; k < iterNum; ++k) {
        caffe::LayerParameter layer = net.layer(1);
        //cout << "\tLayer Type: " << layer.type() << endl;
        caffe::ConvolutionParameter conv;
        caffe::BlobProto blobs, blobs2;
        caffe::BlobShape shape;
        conv = layer.convolution_param();
        blobs = layer.blobs(0);
        blobs2 = layer.blobs(1);
        shape = blobs.shape();

        isPadding = 0;
        isRelu = 0;
        isPoolingMax = 0;
        cfgRow = 24;
        cfgCol = 24;
        cfgM = shape.dim(0);
        cfgN = shape.dim(1);
        cfgK = conv.kernel_size(0);
        cfgS = conv.stride(0);
        cfgPoolK = 0;
        cfgPoolS = 0;
        weightSize = cfgM*cfgN*cfgK*cfgK;
        biasSize = cfgM;
        inputFmSize = cfgN*((cfgRow-1)*cfgS+cfgK)*((cfgCol-1)*cfgS+cfgK);
        outputFmSize = cfgM*cfgRow*cfgCol;
        dram = new float[weightSize+biasSize+inputFmSize+outputFmSize];

        weightOffset = 0;
        biasOffset = weightSize;
        inFmOffset = weightSize + biasSize;
        outFmOffset = weightSize + biasSize + inputFmSize;
        //cout << "FPGA Configuration:" << endl;
        //cout << "isPadding=" << isPadding << endl;
        //cout << "isRelu=" << isRelu << endl;
        //cout << "isPoolingMax=" << isPoolingMax << endl;
        //cout << "cfgRow=" << cfgRow << endl;
        //cout << "cfgCol=" << cfgCol << endl;
        //cout << "cfgM=" << cfgM << endl;
        //cout << "cfgN=" << cfgN << endl;
        //cout << "cfgK=" << cfgK << endl;
        //cout << "cfgS=" << cfgS << endl;
        //cout << "cfgPoolK=" << cfgPoolK << endl;
        //cout << "cfgPoolS=" << cfgPoolS << endl;

        //load input data
        int imgLen = fread(dram+inFmOffset, sizeof(float), inputFmSize, inputfd);

        //load weights
        //cout << "weights num: " << blobs.data_size() << endl;
        for(int i = 0; i < weightSize; ++i)
            *(dram+weightOffset+i) = blobs.data(i);
        //load biases
        //cout << "bias num: " << blobs2.data_size() << endl;
        for(int i = 0; i < biasSize; ++i) 
            *(dram+biasOffset+i) = blobs2.data(i);


        LayerTop(dram, LayerCfgType(weightOffset, biasOffset, inFmOffset, outFmOffset,
                    0, 0, 1, 0,
                    24, 24, 20, 1, 5, 1, 2, 2));

        fd = fopen("conv1.bin", "wb");
        if(fd == NULL) cout << "covn1.bin file not found." << endl;
        fwrite(dram+outFmOffset, sizeof(float), outputFmSize/4, fd);
        fclose(fd);
        delete dram;

        //conv2 load to fpga
        layer = net.layer(3);
        //cout << "\tLayer Type: " << layer.type() << endl;
        conv = layer.convolution_param();
        blobs = layer.blobs(0);
        blobs2 = layer.blobs(1);
        shape = blobs.shape();

        cfgRow = 8;
        cfgCol = 8;
        cfgM = shape.dim(0);
        cfgN = shape.dim(1);
        cfgK = conv.kernel_size(0);
        cfgS = conv.stride(0);
        cfgPoolK = 0;
        cfgPoolS = 0;
        weightSize = cfgM*cfgN*cfgK*cfgK;
        biasSize = cfgM;
        inputFmSize = cfgN*((cfgRow-1)*cfgS+cfgK)*((cfgCol-1)*cfgS+cfgK);
        outputFmSize = cfgM*cfgRow*cfgCol;
        dram = new float[weightSize+biasSize+inputFmSize+outputFmSize];

        weightOffset = 0;
        biasOffset = weightSize;
        inFmOffset = weightSize + biasSize;
        outFmOffset = weightSize + biasSize + inputFmSize;

        //cout << "FPGA Configuration:" << endl;
        //cout << "isPadding=" << isPadding << endl;
        //cout << "isRelu=" << isRelu << endl;
        //cout << "isPoolingMax=" << isPoolingMax << endl;
        //cout << "cfgRow=" << cfgRow << endl;
        //cout << "cfgCol=" << cfgCol << endl;
        //cout << "cfgM=" << cfgM << endl;
        //cout << "cfgN=" << cfgN << endl;
        //cout << "cfgK=" << cfgK << endl;
        //cout << "cfgS=" << cfgS << endl;
        //cout << "cfgPoolK=" << cfgPoolK << endl;
        //cout << "inputFmSize=" << inputFmSize << endl;

        //load input data
        fd = fopen("conv1.bin", "rb");
        if(fd == NULL) cout << "conv1.bin file not found." << endl;
        imgLen = fread(dram+inFmOffset, sizeof(float), inputFmSize, fd);
        fclose(fd);

        //load weights
        //cout << "weights num: " << blobs.data_size() << endl;
        for(int i = 0; i < weightSize; ++i)
            *(dram+weightOffset+i) = blobs.data(i);
        //load biases
        //cout << "bias num: " << blobs2.data_size() << endl;
        for(int i = 0; i < biasSize; ++i) 
            *(dram+biasOffset+i) = blobs2.data(i);
        LayerTop(dram, LayerCfgType(weightOffset, biasOffset, inFmOffset, outFmOffset,
                    0, 0, 1, 0,
                    8, 8, 50, 20, 5, 1, 2, 2));
        fd = fopen("conv2.bin", "ab+");
        if(fd == NULL) cout << "covn2.bin file not found." << endl;
        fwrite(dram+outFmOffset, sizeof(float), outputFmSize/4, fd);
        fclose(fd);
        delete dram;

        if(k%batch != batch-1 && k != iterNum-1) continue;
        if(k == iterNum-1 && iterNum != batch) batch = iterNum%batch;

        //ip3 load to fpga
        layer = net.layer(5);
        //cout << "\tLayer Type: " << layer.type() << endl;
        conv = layer.convolution_param();
        blobs = layer.blobs(0);
        blobs2 = layer.blobs(1);
        shape = blobs.shape();

        cfgRow = 1;
        cfgCol = 1;
        cfgM = 500;
        cfgN = 800;
        cfgK = 1;
        cfgS = 1;
        cfgPoolK = 0;
        cfgPoolS = 0;
        weightSize = cfgM*cfgN*cfgK*cfgK;
        biasSize = cfgM;
        inputFmSize = batch * cfgN*((cfgRow-1)*cfgS+cfgK)*((cfgCol-1)*cfgS+cfgK);
        outputFmSize = batch * cfgM*cfgRow*cfgCol;
        dram = new float[weightSize+biasSize+inputFmSize+outputFmSize];

        weightOffset = 0;
        biasOffset = inputFmSize;
        inFmOffset = inputFmSize + biasSize;
        outFmOffset = inputFmSize + biasSize + weightSize;

        //cout << "FPGA Configuration:" << endl;
        //cout << "isPadding=" << isPadding << endl;
        //cout << "isRelu=" << isRelu << endl;
        //cout << "isPoolingMax=" << isPoolingMax << endl;
        //cout << "cfgRow=" << cfgRow << endl;
        //cout << "cfgCol=" << cfgCol << endl;
        //cout << "cfgM=" << cfgM << endl;
        //cout << "cfgN=" << cfgN << endl;
        //cout << "cfgK=" << cfgK << endl;
        //cout << "cfgS=" << cfgS << endl;
        //cout << "cfgPoolK=" << cfgPoolK << endl;
        //cout << "inputFmSize=" << inputFmSize << endl;

        //load input data
        fd = fopen("conv2.bin", "rb");
        if(fd == NULL) cout << "conv1.bin file not found." << endl;
        imgLen = fread(dram+weightOffset, sizeof(float), inputFmSize, fd);
        fclose(fd);

        //load weights
        //cout << "weights num: " << blobs.data_size() << endl;
        for(int i = 0; i < cfgM; ++i)
            for(int j = 0; j < cfgN; ++j)
                *(dram+inFmOffset+j*cfgM+i) = blobs.data(i*cfgN+j);
        //load biases
        //cout << "bias num: " << blobs2.data_size() << endl;
        for(int i = 0; i < biasSize; ++i) 
            *(dram+biasOffset+i) = blobs2.data(i);
        LayerTop(dram, LayerCfgType(weightOffset, biasOffset, inFmOffset, outFmOffset,
                    0, 1, 0, 1,
                    1, 500, batch, 800, 1, 1, 0, 0));

        fd = fopen("ip3.bin", "wb");
        if(fd == NULL) cout << "ip3.bin file not found." << endl;
        fwrite(dram+outFmOffset, sizeof(float), outputFmSize, fd);
        fclose(fd);
        delete dram;



        //ip4 load to fpga
        layer = net.layer(7);
        //cout << "\tLayer Type: " << layer.type() << endl;
        conv = layer.convolution_param();
        blobs = layer.blobs(0);
        blobs2 = layer.blobs(1);
        shape = blobs.shape();

        cfgRow = 1;
        cfgCol = 1;
        cfgM = 10;
        cfgN = 500;
        cfgK = 1;
        cfgS = 1;
        cfgPoolK = 0;
        cfgPoolS = 0;
        weightSize = cfgM*cfgN*cfgK*cfgK;
        biasSize = cfgM;
        inputFmSize = batch * cfgN*((cfgRow-1)*cfgS+cfgK)*((cfgCol-1)*cfgS+cfgK);
        outputFmSize = batch * cfgM*cfgRow*cfgCol;
        dram = new float[weightSize+biasSize+inputFmSize+outputFmSize];

        weightOffset = 0;
        biasOffset = inputFmSize;
        inFmOffset = inputFmSize + biasSize;
        outFmOffset = inputFmSize + biasSize + weightSize;

        //cout << "FPGA Configuration:" << endl;
        //cout << "isPadding=" << isPadding << endl;
        //cout << "isRelu=" << isRelu << endl;
        //cout << "isPoolingMax=" << isPoolingMax << endl;
        //cout << "cfgRow=" << cfgRow << endl;
        //cout << "cfgCol=" << cfgCol << endl;
        //cout << "cfgM=" << cfgM << endl;
        //cout << "cfgN=" << cfgN << endl;
        //cout << "cfgK=" << cfgK << endl;
        //cout << "cfgS=" << cfgS << endl;
        //cout << "cfgPoolK=" << cfgPoolK << endl;
        //cout << "inputFmSize=" << inputFmSize << endl;

        //load input data
        fd = fopen("ip3.bin", "rb");
        if(fd == NULL) cout << "ip3.bin file not found." << endl;
        imgLen = fread(dram+weightOffset, sizeof(float), inputFmSize, fd);
        fclose(fd);

        //load weights
        //cout << "weights num: " << blobs.data_size() << endl;
        for(int i = 0; i < cfgM; ++i)
            for(int j = 0; j < cfgN; ++j)
                *(dram+inFmOffset+j*cfgM+i) = blobs.data(i*cfgN+j);
        //load biases
        //cout << "bias num: " << blobs2.data_size() << endl;
        for(int i = 0; i < biasSize; ++i) 
            *(dram+biasOffset+i) = blobs2.data(i);
        LayerTop(dram, LayerCfgType(weightOffset, biasOffset, inFmOffset, outFmOffset,
                    0, 0, 0, 1,
                    1, 10, batch, 500, 1, 1, 0, 0));


        //cout << "\toutput:" << endl;
        int index = 0;
        float max, readData;
        int tmpSize = outputFmSize/batch;
        for(int j = 0; j < batch; ++j) {
            for(int i = 0; i < tmpSize; ++i) {
                readData = *(dram+outFmOffset+j*tmpSize+i);
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

            char golden[1];
            int labelLen = fread(golden, sizeof(char), 1, labelfd);
            correcNum += (index == (int)golden[0]);
            cout << k+1-batch+j+1 <<": Calc Number is " << index << "\tExpected is " << ((int)golden[0]) << endl;
            cout << "Accuracy is " << 1.0*correcNum/(k+1-batch+j+1) << endl;
        }
        remove("conv2.bin");



/*

        cout << "\tweights:" << endl;
        for(int i = 0; i < weightSize; ++i) 
            cout << *(dram+weightOffset+i) << endl;

        cout << "\tbiases:" << endl;
        for(int i = 0; i < biasSize; ++i) 
            cout << *(dram+biasOffset+i) << endl;

        cout << "\tinput:" << endl;
        for(int i = 0; i < inputFmSize; ++i) 
            cout << "IMG:" << (i+1) << "\t" << *(dram+inFmOffset+i) << endl;
  
*/            


        delete dram;

    }
    fclose(inputfd);


    return 0;
}



