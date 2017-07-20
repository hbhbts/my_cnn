
#define TM 50
#define TN 7
#define TR 24
#define TC 24
#define K_MAX 5
#define S_MAX 1
#define TR_IN (S_MAX*(TR-1)+K_MAX)
#define TC_IN (S_MAX*(TC-1)+K_MAX)
#define MIN(x, y) (x < y ? x : y)
#define K_POOL_MIN 2
#define S_POOL_MIN 2
#define K_POOL_MAX 2
#define S_POOL_MAX 2
#define TR_POOL ((TR-K_POOL_MIN)/S_POOL_MIN+1)
#define TC_POOL ((TC-K_POOL_MIN)/S_POOL_MIN+1)
#define BIAS_NUM (TM > TC ? TM : TC)


typedef unsigned int ParameterType;
typedef float DataType;
struct LayerCfgType{
    ParameterType weightOffset;
    ParameterType biasOffset;
    ParameterType inFmOffset;
    ParameterType outFmOffset;
    bool isPadding;
    bool isRelu;
    bool isPoolingMax;
    bool isFullConnect;
    ParameterType cfgRow;
    ParameterType cfgCol;
    ParameterType cfgM;
    ParameterType cfgN;
    ParameterType cfgK;
    ParameterType cfgS;
    ParameterType cfgPoolK;
    ParameterType cfgPoolS;

    LayerCfgType(ParameterType weightOffset, ParameterType biasOffset, ParameterType inFmOffset,
            ParameterType outFmOffset, bool isPadding, bool isRelu,
            bool isPoolingMax, bool isFullConnect, ParameterType cfgRow, ParameterType cfgCol,
            ParameterType cfgM, ParameterType cfgN, ParameterType cfgK, 
            ParameterType cfgS, ParameterType cfgPoolK, ParameterType cfgPoolS):
                weightOffset(weightOffset), biasOffset(biasOffset), inFmOffset(inFmOffset), 
                outFmOffset(outFmOffset), isPadding(isPadding), isRelu(isRelu), isFullConnect(isFullConnect),
                isPoolingMax(isPoolingMax), cfgRow(cfgRow), cfgCol(cfgCol), cfgM(cfgM), 
                cfgN(cfgN), cfgK(cfgK), cfgS(cfgS), cfgPoolK(cfgPoolK), cfgPoolS(cfgPoolS) {}

};

//function define
void LayerTop(DataType *dram, DataType *dram2, DataType *dram3, LayerCfgType cfgSet);


