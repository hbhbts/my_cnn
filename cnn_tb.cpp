#include "ap_int.h"
#include <iostream>

using namespace std;

void cnn_top(float*, int, int, int, int, int, int, int, int, int);

int main(int argc, char** argv) {
    int offset_weight;
    int offset_infm;
    int offset_outfm;
    int cfg_row = 13;
    int cfg_col = 13;
    int cfg_n = 192;
    int cfg_m = 128;
    int cfg_k = 3;
    int cfg_s = 1;
    float* dram;
    int weight_size;
    int infm_size;
    int outfm_size;

    weight_size = cfg_m*cfg_n*cfg_k*cfg_k;
    infm_size = cfg_n*(cfg_row*cfg_s)*(cfg_col*cfg_s);
    outfm_size = cfg_m*cfg_row*cfg_col;
    dram = new float[weight_size + infm_size + outfm_size];
    offset_weight = 0;
    offset_infm = weight_size;
    offset_outfm = weight_size + infm_size;
    cout << "weigth size:" << weight_size << "\tinfm size:" << infm_size << "\toutfm_size:"
        << outfm_size << "\ttatoal size:" << weight_size + infm_size + outfm_size << endl;

    cnn_top(dram, offset_weight, offset_infm, offset_outfm, cfg_row, cfg_col, cfg_m , cfg_n, cfg_k, cfg_s);

    return 0;
}



