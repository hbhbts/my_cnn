#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "caffe.pb.h"


using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;



using namespace std;

int main(void) {
    
    const char* filename= "bvlc_alexnet.caffemodel";

    int fd = open(filename, O_RDONLY);

    if(fd == -1) cout << "File not found: " << filename;

    ZeroCopyInputStream* raw_input = new FileInputStream(fd);
    CodedInputStream* coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(2147483647, 536870912);

    
    caffe::NetParameter param;
    bool success = param.ParseFromCodedStream(coded_input);
    if(success) cout << "success parse file" << endl;

    delete coded_input;
    delete raw_input;
    close(fd);

    cout << "net: " << param.name() << endl;
    
    cout << "layer size: " << param.layers_size() << endl;
    for(int i = 0; i < param.layers_size(); ++i) {
        caffe::V1LayerParameter layer = param.layers(i);
        cout << i+1 << ". " << layer.name() << endl;
        const caffe::ConvolutionParameter conv = layer.convolution_param();
        const caffe::PoolingParameter pool = layer.pooling_param();
        const caffe::ReLUParameter relu = layer.relu_param();
        const caffe::InnerProductParameter ip = layer.inner_product_param();
        switch (layer.type()) {
            case caffe::V1LayerParameter::CONVOLUTION:
                cout << "\tConv:" << endl;
                cout << "\t\tnum_output: " << conv.num_output() << endl;
                if(conv.pad_size())
                    cout << "\t\tpad: " << conv.pad(0) << endl;
                else 
                    cout << "\t\tpad: 0" << endl;

                if(conv.kernel_size_size())
                    cout << "\t\tkernel: " << conv.kernel_size(0) << endl;
                else 
                    cout << "\t\tkernel: 1" << endl;

                if(conv.stride_size())
                    cout << "\t\tstride: " << conv.stride(0) << endl;
                else 
                    cout << "\t\tstride: 1" << endl;
                cout << "\tData:" << endl;
                for(int i = 0; i < layer.blobs_size(); ++i) {
                    const caffe::BlobProto proto = layer.blobs(i);
                    if(i == 0) 
                        cout << "\t\tWeights:" << endl;
                    else
                        cout << "\t\tBiases:" << endl;
                    cout << "\t\tnum: " << proto.num() << endl;
                    cout << "\t\tchannels: " << proto.channels() << endl;
                    cout << "\t\theight: " << proto.height() << endl;
                    cout << "\t\twidth: " << proto.width() << endl;
                    cout << "\t\tdata size: " << proto.data_size() << endl;
                    cout << endl;
                }
                break;
            case caffe::V1LayerParameter::POOLING:
                cout << "\tPool: ";
                if(pool.pool() == caffe::PoolingParameter::MAX)
                    cout << "MAX" << endl;
                else if(pool.pool() == caffe::PoolingParameter::AVE)
                    cout << "AVE" << endl;
                else if(pool.pool() == caffe::PoolingParameter::STOCHASTIC)
                    cout << "STOCHASTIC" << endl;
                cout << "\t\tpad: " << pool.pad() << endl;
                cout << "\t\tkernel_size: " << pool.kernel_size() << endl;
                cout << "\t\tstride: " << pool.stride() << endl;
                break;
            case caffe::V1LayerParameter::RELU:
                cout << "\tRelu: " << endl;
                cout << "\t\tnegative_slope: " << relu.negative_slope() << endl;
                break;
            case caffe::V1LayerParameter::INNER_PRODUCT:
                cout << "\tInnerProduct: " << endl;
                for(int i = 0; i < layer.blobs_size(); ++i) {
                    const caffe::BlobProto proto = layer.blobs(i);
                    if(i == 0) 
                        cout << "\t\tWeights:" << endl;
                    else
                        cout << "\t\tBiases:" << endl;
                    if(i == 0) {
                        cout << "\t\tnum_input: " << proto.width() << endl;
                        cout << "\t\tnum_output: " << proto.height() << endl;
                    }
                    else 
                        cout << "\t\tnum_output: " << proto.width() << endl;
                        
                    cout << "\t\tdata size: " << proto.data_size() << endl;
                    cout << endl;
                }
                break;
            default: 
                cout << "\tundefined" << endl;

        }


    }


}


