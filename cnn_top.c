#include <stdio.h>
#include <ap_cint.h>


#define AXI_BUS_WIDTH 32




typedef struct layer {
    int channels_in;
    int height_in;
    int width_in;
    int channels_out;
    int kernel_size;
    uint<AXI_BUS_WIDTH> input_kernel_addr;
    int stride;
    int pad;
    uint<AXI_BUS_WIDTH> input_data_addr;
    uint<AXI_BUS_WIDTH> output_data_addr;
    /* some bool configurations */
    uint<1> is_relu;
    uint<1> is_first_split;
    uint<1> is_second_split;
} layer_t;

typedef struct network {
    int layers_num;
    layer_t *layers_p;
    int kernels_num;
    float *kernels_p;
    int datas_num;
    float *datas_p;
} network_t;


void fpga_top (uint<AXI_BUS_WIDTH> *main_mem_p, layer_t input_layer) {


}






