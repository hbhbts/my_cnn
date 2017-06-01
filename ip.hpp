
#define TM 64
#define TN 7
#define TR 13
#define TC 13
#define K_MAX 11
#define S_MAX 4
#define TR_IN (S_MAX*(TR-1)+K_MAX)
#define TC_IN (S_MAX*(TC-1)+K_MAX)
#define MIN(x, y) (x < y ? x : y)
#define TR_POOL 6
#define TC_POOL 6


typedef int ParameterType;
typedef float DataType;
struct LayerCfgType{
    ParameterType weightOffset;
    ParameterType biasOffset;
    ParameterType inFmOffset;
    ParameterType outFmOffset;
    ParameterType isPadding;
    ParameterType isRelu;
    ParameterType isPoolingMax;
    ParameterType cfgRow;
    ParameterType cfgCol;
    ParameterType cfgM;
    ParameterType cfgN;
    ParameterType cfgK;
    ParameterType cfgS;
    ParameterType cfgPoolK;
    ParameterType cfgPoolS;

    LayerCfgType(ParameterType weightOffset, ParameterType biasOffset, ParameterType inFmOffset,
            ParameterType outFmOffset, ParameterType isPadding, ParameterType isRelu,
            ParameterType isPoolingMax, ParameterType cfgRow, ParameterType cfgCol,
            ParameterType cfgM, ParameterType cfgN, ParameterType cfgK, 
            ParameterType cfgS, ParameterType cfgPoolK, ParameterType cfgPoolS):
                weightOffset(weightOffset), biasOffset(biasOffset), inFmOffset(inFmOffset), 
                outFmOffset(outFmOffset), isPadding(isPadding), isRelu(isRelu), 
                isPoolingMax(isPoolingMax), cfgRow(cfgRow), cfgCol(cfgCol), cfgM(cfgM), 
                cfgN(cfgN), cfgK(cfgK), cfgS(cfgS), cfgPoolK(cfgPoolK), cfgPoolS(cfgPoolS) {}

};

//function define
void PE(ParameterType isRelu, ParameterType cfgK, ParameterType kernelPos,
            ParameterType trrInPos, ParameterType tccInPos, 
            ParameterType trrOutPos, ParameterType tccOutPos);

void TileConv(ParameterType isRelu, ParameterType cfgK, ParameterType cfgS);

void TilePooling(ParameterType isPoolingMax, 
                    ParameterType poolK, ParameterType poolS);

void TileWriteBack(DataType *dram, ParameterType outFmOffset, ParameterType cfgRow,
        ParameterType cfgCol, ParameterType cfgM, ParameterType row, ParameterType col,
        ParameterType co);

void InputTileLoad(DataType *dram, ParameterType isPadding, ParameterType inFmOffset, ParameterType cfgN,
                    ParameterType cfgK, ParameterType cfgS, ParameterType cfgRow, 
                    ParameterType cfgCol, ParameterType row, ParameterType col,
                    ParameterType ci);

void WeightTileLoad(DataType *dram, ParameterType weightOffset, ParameterType cfgK, 
                    ParameterType cfgN, ParameterType cfgM, ParameterType co, 
                    ParameterType ci);

void BiasTileLoad(DataType *dram, ParameterType biasOffset, ParameterType co);

void LayerTop(DataType *dram, LayerCfgType cfgSet);


