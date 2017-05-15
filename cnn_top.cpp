#include <iostream>
#include "cnn_top.hpp"


using namespace std;

static float weight_buffer[TM][TN][K*K];
static float input_fm_buffer[TN][(TR-1)*S+K][(TC-1)*S+K];
static float imm_acc[TR][TC][TM];

void cnn_top(float *dram, layer_cfg_t layer_in) {
#pragma HLS interface m_axi port=dram depth=128 offset=slave bundle=BUS0
#pragma HLS interface s_axilite port=return bundle=BUS0
#pragma HLS interface s_axilite port=layer_in bundle=BUS0


	int offset_weight, offset_infm, offset_outfm, cfg_row, cfg_col, cfg_m, cfg_n, cfg_k, cfg_s;
    int row, col, co, ci, i, j, trr, tcc, too, tii;
    int lwi, lwj, lwk;
    int lini, linj, link, linp, linq;


    offset_weight = layer_in.offset_weight;
    offset_infm = layer_in.offset_infm;
    offset_outfm = layer_in.offset_outfm;
    cfg_row = layer_in.cfg_row;
    cfg_col = layer_in.cfg_col;
    cfg_m = layer_in.cfg_m;
    cfg_n = layer_in.cfg_n;
    cfg_k = layer_in.cfg_k;
    cfg_s = layer_in.cfg_s;
    
    for(row = 0; row < cfg_row; row += TR) {
        for(col = 0; col < cfg_col;  col += TC) {
            for(co = 0; co < cfg_m; co += TM) {
                for(ci = 0; ci < cfg_n; ci += TN) {
                    cout << "row:" << row << "\tcol:" << col << "\tco:" << co << "\tci:" << ci << endl;

                    //load weights
                    load_weights(dram, offset_weight, cfg_k, cfg_n, cfg_m, co, ci);

                    //load input feature maps
                    load_inputfm(dram, offset_infm, cfg_n, cfg_k, cfg_s,  cfg_col, cfg_row, row, col, ci);

                    //convolutional calc
                    conv_calc(dram, offset_outfm, cfg_k, cfg_s, cfg_col, cfg_row, cfg_m, cfg_n, row, col, ci, co);

                }
LOOP_INDEX:
	            for(int ltrr = 0; ltrr < TR; ++ltrr) {
		            for(int ltcc = 0; ltcc < TC; ++ltcc) {
			            for(int ltoo = 0; ltoo < TM; ++ltoo) {
#pragma HLS unroll
				            if(ltrr+row < cfg_row && ltcc+col < cfg_col && too+co < cfg_m) {
					            *(dram + offset_outfm + ((too+co)*TR*TC+(ltrr+row)*TR+ltcc+col)) = imm_acc[ltrr][ltcc][ltoo];
				            }
			            }
		            }
                }

            }
        }
    }

}


void conv_calc(float* dram, int offset_outfm, int cfg_k, int cfg_s, int cfg_col, int cfg_row, int cfg_m, int cfg_n, int row, int col, int ci, int co) {
#pragma HLS inline
int i, j, trr, tcc, too, tii;
int ltrr, ltcc, ltoo;
#pragma HLS array_partition variable=imm_acc complete dim=3
#pragma HLS resource variable=imm_acc core=RAM_S2P_BRAM


LOOP_J:
	for(i = 0; i < cfg_k; ++i) {
LOOP_I:
		for(j = 0; j < cfg_k; ++j) {
LOOP_TRR:
			for(trr = 0; trr < TR; ++trr) {
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

							imm_reg = weight_buffer[too][tii][i*cfg_k+j] * input_fm_buffer[tii][cfg_s*trr+i][cfg_s*tcc+j];
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


void load_weights(float* dram, int offset_weight, int cfg_k, int cfg_n, int cfg_m, int co, int ci) {
#pragma HLS inline

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
                        } else {
                        	input_fm_buffer[lini][linj*cfg_s+linp][link*cfg_s+linq] = 0;
                        }
                    }
                }
            }
        }
    }
}
