#include <iostream>
#include "ap_int.h"
#include "cnn_top.hpp"
#include "assert.h"


using namespace std;

static float weight_buffer[TM][TN][K*K];
static float input_fm_buffer[TN][(TR-1)*S+K-1][(TC-1)*S+K-1];
static float imm_acc[TR][TC][TM];


void cnn_top(float *dram, float *dram2, float *dram3, layer_cfg_t layer_in) {
#pragma HLS interface m_axi port=dram depth=128 offset=slave bundle=BUS0
#pragma HLS interface m_axi port=dram2 depth=128 offset=slave bundle=BUS1
#pragma HLS interface m_axi port=dram3 depth=128 offset=slave bundle=BUS0
#pragma HLS interface s_axilite port=return bundle=BUS0
#pragma HLS interface s_axilite port=layer_in bundle=BUS0


	int offset_weight, offset_infm, offset_outfm, cfg_row, cfg_col, cfg_m, cfg_n, cfg_k, cfg_s;
    int row, col, co, ci, i, j, trr, tcc, too, tii;
    int lwi, lwj, lwk;
    int lini, linj, link, linp, linq;


    assert(layer_in.cfg_n <= N);
    assert(layer_in.cfg_m <= M);
    assert(layer_in.cfg_row <= ROW);
    assert(layer_in.cfg_col <= COL);
    assert(layer_in.cfg_k <= K);
    assert(layer_in.cfg_s <= S);


    offset_weight = layer_in.offset_weight;
    offset_infm = layer_in.offset_infm;
    offset_outfm = layer_in.offset_outfm;
    cfg_row = layer_in.cfg_row;
    cfg_col = layer_in.cfg_col;
    cfg_m = layer_in.cfg_m;
    cfg_n = layer_in.cfg_n;
    cfg_k = layer_in.cfg_k;
    cfg_s = layer_in.cfg_s;

LOOP_MAIN:
    for(row = 0; row < cfg_row; row += TR) {
        for(col = 0; col < cfg_col;  col += TC) {
            for(co = 0; co < cfg_m; co += TM) {
                for(ci = 0; ci < cfg_n; ci += TN) {
//pragma HLS dataflow
                    cout << "row:" << row << "\tcol:" << col << "\tco:" << co << "\tci:" << ci << endl;

                    cnn_core(dram, dram2, dram3, offset_weight, offset_infm, offset_outfm, cfg_row, cfg_col, cfg_n, cfg_m, cfg_k, cfg_s, row, col, co, ci);

                }
            }
        }
    }

}

void cnn_core(float *dram, float *dram2, float *dram3, int offset_weight, int offset_infm, int offset_outfm, int cfg_row, int cfg_col, int cfg_n, int cfg_m, int cfg_k, int cfg_s, int row, int col, int co, int ci) {
#pragma HLS inline
    //load weights
	load_weights(dram, offset_weight, cfg_k, cfg_n, cfg_m, co, ci);

    //load input feature maps
    load_inputfm(dram2, offset_infm, cfg_n, cfg_k, cfg_s,  cfg_col, cfg_row, row, col, ci);

    //convolutional calc
    conv_calc(cfg_k, cfg_s);

    store_outputfm(dram3, offset_outfm, cfg_row, cfg_col, cfg_m, row, col, co);
}

void conv_calc(int cfg_k, int cfg_s) {
#pragma HLS inline
int i, j, trr, tcc, too, tii;
int ltrr, ltcc, ltoo;

#pragma HLS array_partition variable=imm_acc complete dim=3
#pragma HLS resource variable=imm_acc core=RAM_S2P_BRAM




LOOP_J:
	for(i = 0; i < cfg_k; ++i) {
LOOP_I:
		for(j = 0; j < cfg_k; ++j) {
			int new_k;
			{
#pragma HLS latency min=1
			new_k = i*cfg_k+j;
			}
LOOP_TRR:
			for(trr = 0; trr < TR; ++trr) {
				int new_trr;
				{
#pragma HLS latency min=1
				new_trr = cfg_s*trr+i;
				}
LOOP_TCC:
				for(tcc = 0; tcc < TC; ++tcc) {
#pragma HLS pipeline II=1
LOOP_TOO:
					for(too = 0; too < TM; ++too) {
#pragma HLS unroll
						float imm_reg2;

						for(tii = 0; tii < TN; ++tii) {
#pragma HLS unroll
							float imm_reg;

							imm_reg = weight_buffer[too][tii][new_k] * input_fm_buffer[tii][new_trr][cfg_s*tcc+j];
	                        if(tii == 0)
	                        	imm_reg2 = imm_reg;
	                        else
	                        	imm_reg2 += imm_reg;
	                    }

						imm_acc[trr][tcc][too] = imm_reg2;
	                }
				}
			}
	   }
	}


}


void store_outputfm(float* dram, int offset_outfm, int cfg_row, int cfg_col, int cfg_m, int row, int col, int co) {
#pragma HLS inline
	int ltrr, ltcc, ltoo;

LOOP_INDEX:
    for(ltoo = co; ltoo < MIN(co+TM, cfg_m); ++ltoo) {
    	for(ltrr = row; ltrr < MIN(row+TR, cfg_row); ++ltrr) {
    		for(ltcc = 0; ltcc < TC; ++ltcc) {
#pragma HLS pipeline
    			if(ltcc+col < cfg_col)
    				*(dram + offset_outfm + ltoo*(ltrr*cfg_col+col+ltcc)) = imm_acc[ltrr-row][ltcc][ltoo-co];
    		}
    	}
    }


}

void load_weights(float* dram, int offset_weight, int cfg_k, int cfg_n, int cfg_m, int co, int ci) {
#pragma HLS inline

#pragma HLS array_partition variable=weight_buffer complete dim=1
#pragma HLS array_partition variable=weight_buffer complete dim=2
#pragma HLS resource variable=weight_buffer core=RAM_S2P_BRAM

    int lwi, lwj, lwk;
    int cfg_k_sqr = cfg_k*cfg_k;
LOOP_LW:
    for(lwi = co; lwi < MIN(co+TM, cfg_m); ++lwi) {
        for(lwj = ci; lwj < MIN(ci+TN, cfg_n); ++lwj) {
            for(lwk = 0; lwk < cfg_k_sqr; ++lwk) {
#pragma HLS pipeline
                if(co+lwi < cfg_m && ci+lwj < cfg_n) {
                    weight_buffer[lwi-co][lwj-ci][lwk] = *(dram + offset_weight + (lwi*TN+lwj)*cfg_k_sqr + lwk);
                } else {
                	weight_buffer[lwi][lwj][lwk] = 0;
                }
            }
        }
    }
}


void load_inputfm(float* dram, int offset_infm, int cfg_n, int cfg_k, int cfg_s, int cfg_col, int cfg_row, int row, int col, int ci) {
#pragma HLS inline

#pragma HLS array_partition variable=input_fm_buffer complete dim=1    
#pragma HLS resource variable=input_fm_buffer core=RAM_S2P_BRAM
int lini, linj, link, linp, linq;


LOOP_LINP:
    for(lini = 0; lini < TN; ++lini) {
        for(linj = 0; linj < TR*cfg_s+cfg_k-1; ++linj) {
            for(link = 0; link < TC*S+K-1; ++link) {
#pragma HLS pipeline
            	if(link < TC*cfg_s+cfg_k-1)
            		input_fm_buffer[lini][linj][link] =
                                *(dram + offset_infm + (lini+ci)*cfg_row*cfg_col + (linj+row)*cfg_col + link);

            }
        }
    }
}
