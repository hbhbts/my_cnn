#include "ap_int.h"
#include <iostream>

#define TM 67
#define TN 7
#define TR 13
#define TC 13
#define S 4
#define K 11

using namespace std;

void load_weights(float* dram, int offset_weight, int cfg_k, int cfg_n, int cfg_m, int co, int ci);
void load_inputfm(float* dram, int offset_infm, int cfg_n, int cfg_k, int cfg_s, int cfg_col, int cfg_row, int row, int col, int ci);

static float weight_buffer[TM][TN][K*K];
static float input_fm_buffer[TN][(TR-1)*S+K][(TC-1)*S+K];
    

void cnn_top(float *dram, int offset_weight, int offset_infm, int offset_outfm, int cfg_row, int cfg_col, int cfg_m , int cfg_n, int cfg_k, int cfg_s) {
#pragma HLS interface m_axi port=dram depth=256 offset=slave bundle=BUS0
#pragma HLS interface s_axilite port=return bundle=BUS0
#pragma HLS interface s_axilite port=offset_weight bundle=BUS0
#pragma HLS interface s_axilite port=offset_infm bundle=BUS0
#pragma HLS interface s_axilite port=offset_outfm bundle=BUS0
#pragma HLS interface s_axilite port=cfg_row bundle=BUS0
#pragma HLS interface s_axilite port=cfg_col bundle=BUS0
#pragma HLS interface s_axilite port=cfg_m bundle=BUS0
#pragma HLS interface s_axilite port=cfg_n bundle=BUS0
#pragma HLS interface s_axilite port=cfg_k bundle=BUS0
#pragma HLS interface s_axilite port=cfg_s bundle=BUS0

    int row, col, co, ci, i, j, trr, tcc, too, tii;
    int lwi, lwj, lwk;
    int lini, linj, link, linp, linq;
    static float imm_mem[TM];
    float imm_reg;
#pragma HLS array_partition variable=imm_mem complete
    
    for(row = 0; row < cfg_row; row += TR) {
        for(col = 0; col < cfg_col;  col += TC) {
            for(co = 0; co < cfg_m; co += TM) {
                for(ci = 0; ci < cfg_n; ci += TN) {
                    cout << "row:" << row << "\tcol:" << col << "\tco:" << co << "\tci:" << ci << endl;

                    //load weights
                    load_weights(dram, offset_weight, cfg_k, cfg_n, cfg_m, co, ci);

                    //load input feature maps
                    load_inputfm(dram, offset_infm, cfg_n, cfg_k, cfg_s,  cfg_col, cfg_row, row, col, ci);

                    //convalutional calc
LOOP_J:
                    for(i = 0; i < cfg_k; ++i) {
LOOP_I:
                        for(j = 0; j < cfg_k; ++j) {
LOOP_TRR:
                            for(trr = row; trr < row+TR; ++trr) {
LOOP_TCC:
                                for(tcc = col; tcc < col+TC; ++tcc) {
#pragma HLS pipeline enable_flush
LOOP_TOO:
                                    for(too = 0; too < TM; ++too) {
#pragma HLS unroll                       
LOOP_TII:
                                        for(tii = 0; tii < TN; ++tii) {
#pragma HLS unroll
                                            if(trr+i < cfg_row && tcc+j < cfg_col && tii+ci < cfg_n && too+co < cfg_m) {
                                                //cout << "weight:" << weight_buffer[too][tii][i*cfg_k+j] << endl;
                                                //cout << "inputfm:" << input_fm_buffer[tii][cfg_s*(trr-row)+i][cfg_s*(tcc-col)+j] << endl;
                                                //cout << "addr:" << offset_outfm + ((too+co)*TR*TC+trr*TR+tcc) << endl;
                                                //cout << "outputfm:" << *((float*)(dram + offset_outfm + ((too+co)*TR*TC+trr*TR+tcc))) << endl;
                                                imm_reg = weight_buffer[too][tii][i*cfg_k+j] * input_fm_buffer[tii][cfg_s*(trr-row)+i][cfg_s*(tcc-col)+j];
                                                if(too == 0 && tii == 0)
                                                    imm_mem[too] = imm_reg;
                                                else 
                                                    imm_mem[too] += imm_reg;

                                            }
                                        }
                                    }
LOOP_INDEX:
                                    for(int index = 0; index < TM; ++index) {
#pragma HLS unroll
                                        *(dram + offset_outfm + ((too+co)*TR*TC+trr*TR+tcc)) = imm_mem[index];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


}


void load_weights(float* dram, int offset_weight, int cfg_k, int cfg_n, int cfg_m, int co, int ci) {

#pragma HLS array_partition variable=weight_buffer complete dim=1
#pragma HLS array_partition variable=weight_buffer complete dim=2
#pragma HLS resource variable=weight_buffer core=RAM_S2P_BRAM

    int lwi, lwj, lwk;

    for(lwi = 0; lwi < TM; ++lwi) {
        for(lwj = 0; lwj < TN; ++lwj) {
LOOP_LWJ:
            for(lwk = 0; lwk < cfg_k*cfg_k; ++lwk) {
#pragma HLS pipeline
                if(co+lwi < cfg_m && ci+lwj < cfg_n) {
                    weight_buffer[lwi][lwj][lwk] = *(dram + offset_weight + (lwi*TM+lwj)*cfg_k*cfg_k + lwk);
                    //cout << "w[" << lwi << "][" << lwj << "][" << lwk << "]" << endl;
                }
            }
        }
    }
}


void load_inputfm(float* dram, int offset_infm, int cfg_n, int cfg_k, int cfg_s, int cfg_col, int cfg_row, int row, int col, int ci) {

#pragma HLS array_partition variable=input_fm_buffer complete dim=1    
#pragma HLS resource variable=input_fm_buffer core=RAM_S2P_BRAM
    int lini, linj, link, linp, linq;

    for(lini = 0; lini < TN; ++lini) {
        for(linj = 0; linj < TR; ++linj) {
            for(link = 0; link < TC; ++link) {
                for(linp = 0; linp < cfg_k; ++linp) {
LOOP_LINP:
                    for(linq = 0; linq < cfg_k; ++linq) {
#pragma HLS pipeline                                            
                        if(lini+ci < cfg_n && linj+row+linp < cfg_row && link+col+linq < cfg_col) {
                            input_fm_buffer[lini][linj*cfg_s+linp][link*cfg_s+linq] = 
                                *(dram + offset_infm + (((lini*TN+linj)*TR+link)*TC+linp)*cfg_k + cfg_k);
                            //cout << "ifm[" << lini << "][" << linj*cfg_s+linp << "][" << link*cfg_s+linq << "]" << endl;
                        }
                    }
                }
            }
        }
    }
}
