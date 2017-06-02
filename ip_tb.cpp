#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <stdio.h>
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
    FILE *fd;

    caffe::NetParameter net;
    const char weightsName[] = "lenet_iter_10000.caffemodel";
    bool success = ReadProtoFromBinaryFile(weightsName, &net);
    cout << "Net Name: " << net.name() << endl;
    cout << "Layers Num: " << net.layer_size() << endl;

    if(net.layer_size() > 0) {
        caffe::LayerParameter layer = net.layer(1);
        cout << "\tLayer Type: " << layer.type() << endl;
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
        cout << "FPGA Configuration:" << endl;
        cout << "isPadding=" << isPadding << endl;
        cout << "isRelu=" << isRelu << endl;
        cout << "isPoolingMax=" << isPoolingMax << endl;
        cout << "cfgRow=" << cfgRow << endl;
        cout << "cfgCol=" << cfgCol << endl;
        cout << "cfgM=" << cfgM << endl;
        cout << "cfgN=" << cfgN << endl;
        cout << "cfgK=" << cfgK << endl;
        cout << "cfgS=" << cfgS << endl;
        cout << "cfgPoolK=" << cfgPoolK << endl;
        cout << "cfgPoolS=" << cfgPoolS << endl;

        //load input data
        fd = fopen("img.bin", "rb");
        if(fd == NULL) cout << "img.bin file not found." << endl;
        int imgLen = fread(dram+inFmOffset, sizeof(float), inputFmSize, fd);
        fclose(fd);

        //load weights
        cout << "weights num: " << blobs.data_size() << endl;
        for(int i = 0; i < weightSize; ++i)
            *(dram+weightOffset+i) = blobs.data(i);
        //load biases
        cout << "bias num: " << blobs.diff_size() << endl;
        for(int i = 0; i < biasSize; ++i) 
            *(dram+biasOffset+i) = blobs2.data(i);


        LayerTop(dram, LayerCfgType(weightOffset, biasOffset, inFmOffset, outFmOffset,
                    0, 0, 0, 
                    24, 24, 20, 1, 5, 1, 2, 2));



        cout << "dram: " << endl;
        cout << "\tweights:" << endl;
        for(int i = 0; i < 0; ++i) 
            cout << *(dram+weightOffset+i) << endl;

        cout << "\tbiases:" << endl;
        for(int i = 0; i < 0; ++i) 
            cout << *(dram+biasOffset+i) << endl;

        cout << "\tinput:" << endl;
        for(int i = 0; i < 0; ++i) 
            cout << "IMG:" << (i+1) << "\t" << *(dram+inFmOffset+i) << endl;
            
        cout << "\toutput:" << endl;
        for(int i = 0; i < outputFmSize; ++i) 
            cout << (i+1) << "\t\t" << *(dram+outFmOffset+i) << endl;


        delete dram;

    }


    return 0;
}



