
#define TM 64
#define TN 7
#define TR 13
#define TC 13

#define N 192
#define M 128
#define ROW 13
#define COL 13
#define S 1
#define K 3

#define MIN(x, y) (x < y ? x : y)

typedef struct layer_cfg{
	int offset_weight;
	int offset_infm;
	int offset_outfm;
	int cfg_row;
	int cfg_col;
	int cfg_m;
	int cfg_n;
	int cfg_k;
	int cfg_s;

	layer_cfg(int offset_weight, int offset_infm, int offset_outfm, int cfg_row, int cfg_col, int cfg_m , int cfg_n, int cfg_k, int cfg_s):
				offset_weight(offset_weight), offset_infm(offset_infm), offset_outfm(offset_outfm), cfg_row(cfg_row),
				cfg_col(cfg_col), cfg_m(cfg_m), cfg_n(cfg_n), cfg_k(cfg_k), cfg_s(cfg_s) {};
}layer_cfg_t;

void conv_calc(int cfg_k, int cfg_s);
void load_weights(float* dram, int offset_weight, int cfg_k, int cfg_n, int cfg_m, int co, int ci);
void load_inputfm(float* dram, int offset_infm, int cfg_n, int cfg_k, int cfg_s, int cfg_col, int cfg_row, int row, int col, int ci);
void store_outputfm(float* dram, int offset_outfm, int cfg_row, int cfg_col, int cfg_m, int row, int col, int co);
void cnn_core(float *dram, float *dram2, float *dram3, int offset_weight, int offset_infm, int offset_outfm, int cfg_row, int cfg_col, int cfg_n, int cfg_m, int cfg_k, int cfg_s, int row, int col, int co, int ci);
