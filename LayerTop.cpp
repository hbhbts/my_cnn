#include <iostream>
#include <assert.h>
#include "LayerTop.hpp"

using namespace std;


static DataType biasBuffer[BIAS_NUM];
static DataType weightBuffer[TM][TN][K_MAX*K_MAX];
static DataType inputFmBuffer[TN][TR_IN][TC_IN];
static DataType immBuffer[2][TM][TR][TC];
static DataType immBuffer2[TM][TR][TC];
static DataType immPoolArray[TM];
static DataType immPoolBuffer[TM][TR][TC];


void elementPE(bool isRelu, bool isFullConnect, ParameterType cfgN, ParameterType cfgK,
                    int ci, int trr, int tcc, int kernelPos, int trrInPos, int tccInPos, bool sel) {
#pragma HLS inline off
#pragma HLS resource variable=immBuffer core=RAM_2P_BRAM
#pragma HLS array_partition variable=immBuffer dim=1 complete
#pragma HLS array_partition variable=immBuffer dim=2 complete
//#pragma HLS dependence variable=immBuffer intra false
#pragma HLS resource variable=immBuffer2 core=RAM_2P_BRAM  
#pragma HLS array_partition variable=immBuffer2 dim=1 complete

    for(int too = 0; too < TM; ++too) {
#pragma HLS unroll
        DataType immReg = 0;
        for(int tii = 0; tii < TN; ++tii) {
#pragma HLS unroll
            immReg += weightBuffer[too][tii][kernelPos] *
                                inputFmBuffer[tii][trrInPos][tccInPos];
        }

        int biasBufferIndex = isFullConnect ? tcc : too;
        DataType resultAdd = immReg + immBuffer[!sel][too][trr][tcc];
        if(ci == 0 && kernelPos == 0) 
        	immBuffer[sel][too][trr][tcc] = biasBuffer[biasBufferIndex] + immReg;
        else if(ci+TN >= cfgN && kernelPos == cfgK*cfgK-1) {
            DataType localReg = isRelu ? (resultAdd > 0 ? resultAdd : 0) : resultAdd;
            immBuffer2[too][trr][tcc] = localReg;
        } else
            immBuffer[sel][too][trr][tcc] = resultAdd;
    }
}


void PE(bool isRelu, bool isFullConnect, ParameterType cfgRow, ParameterType cfgN, ParameterType cfgK,
            ParameterType cfgS, int ci, int i, int j, bool sel) {
#pragma HLS inline off
    for(int trr = 0; trr < cfgRow; ++trr) {
        for(int tcc = 0; tcc < TC; ++tcc) {
#pragma HLS pipeline 
            int kernelPos = i*cfgK + j;
            int trrInPos = trr * cfgS + i;
            int tccInPos = tcc * cfgS + j;

            elementPE(isRelu, isFullConnect, cfgN, cfgK, ci, trr, tcc, kernelPos, trrInPos, tccInPos, sel);

        }
    }
}


void TileConv(bool isRelu, bool isFullConnect, ParameterType cfgRow, ParameterType cfgN, ParameterType cfgK, ParameterType cfgS, int ci) {
#pragma HLS inline off
    static bool sel = 0;
    for(int i = 0; i < cfgK; ++i) {
        for(int j = 0; j < cfgK; ++j) {
            if(ci == 0 && i == 0 && j == 0)
                sel = 0;
            else 
                sel = !sel;
            	PE(isRelu, isFullConnect, cfgRow, cfgN, cfgK, cfgS, ci, i, j, sel);
        }
    }
}

void TilePooling(ParameterType cfgN, ParameterType cfgK, ParameterType poolK, ParameterType poolS) {
#pragma HLS inline off
#pragma HLS array_partition variable=immPoolArray dim=1 complete
#pragma HLS resource variable=immPoolBuffer core=RAM_2P_BRAM
#pragma HLS array_partition variable=immPoolBuffer dim=1 complete
    ParameterType trrOut = (TR-poolK)/poolS+1; //need to be integer
    ParameterType tccOut = (TC-poolK)/poolS+1; //need to be integer
    //bool sel = ((cfgN/TN)%2) && (cfgN%TN == 0) ^ ((cfgK*cfgK)%2);
    //cout << "pool: " << sel << endl;
    for(int trr = 0; trr < trrOut; ++trr) {
        for(int tcc = 0; tcc < tccOut; ++tcc) {
            for(int i = 0; i < poolK; ++i) {
                for(int j = 0; j < K_POOL_MAX; ++j) {
#pragma HLS pipeline
                    if(j < poolK) {
                        for(int too = 0; too < TM; ++too) {
#pragma HLS unroll
                            if(i == 0 && j == 0)
                                immPoolArray[too] = immBuffer2[too][trr*poolS+i][tcc*poolS+j];
                            else
                                immPoolArray[too] = immBuffer2[too][trr*poolS+i][tcc*poolS+j]
                                                    > immPoolArray[too] ?
                                                    immBuffer2[too][trr*poolS+i][tcc*poolS+j]
                                                    : immPoolArray[too];
                            //if(too == 0)
                            //    cout << "LOOP: " << immBuffer[too][trr*poolS+i][tcc*poolS+j] << endl;
                            immPoolBuffer[too][trr][tcc] = immPoolArray[too];
                            //if(too == 0)
                            //    cout << "POOL: " << too << "\t" << trr << "\t" << tcc << "\t" << immPoolArray[too] << endl;
                        }
                    }
                }
            }
        }
    }
}



void TileWriteBack(DataType *dram, ParameterType outFmOffset, ParameterType cfgRow,
        ParameterType cfgCol, ParameterType cfgM, ParameterType row, ParameterType col,
        ParameterType co, ParameterType tileTR, ParameterType tileTC) {
#pragma HLS inline
    for(int too = 0; too < MIN(TM,cfgM-co); ++too) {
        for(int trr = 0; trr < MIN(tileTR, cfgRow-row); ++ trr) {
            for(int tcc = 0; tcc < TC; ++ tcc) {
#pragma HLS pipeline
                if(tcc < MIN(tileTC, cfgCol-col)) {
                    DataType readReg;
                    readReg = immPoolBuffer[too][trr][tcc];
                    ParameterType immOffset = outFmOffset + (too+co)*cfgRow*cfgCol
                                                + (trr+row)*cfgCol + col + tcc;
                    *(dram + immOffset) = readReg;
                }
                //if(too == 0)
                //    cout << "WB:" << too << "\t" << trr << "\t" << tcc << "\t" << (immOffset-outFmOffset+1) << ": " << readReg << endl;
            }
        }
    }
}

void TileWriteBackWrapper(DataType *dram, ParameterType outFmOffset, bool isPoolingMax, ParameterType cfgRow,
        ParameterType cfgCol, ParameterType cfgM, ParameterType cfgPoolK, ParameterType cfgPoolS, 
        ParameterType row, ParameterType col, ParameterType co) {
#pragma HLS inline off

    int cfgRowTmp, cfgColTmp, rowTmp, colTmp, tileTR, tileTC;

    if(isPoolingMax) {
        cfgRowTmp = (cfgRow-cfgPoolK)/cfgPoolS + 1;
        cfgColTmp = (cfgCol-cfgPoolK)/cfgPoolS + 1;
        rowTmp = (row/cfgPoolK);
        colTmp = (col/cfgPoolK);
        tileTR = (TR-cfgPoolK)/cfgPoolS + 1;
        tileTC = (TC-cfgPoolK)/cfgPoolS + 1;
    } else {
        cfgRowTmp = cfgRow;
        cfgColTmp = cfgCol;
        rowTmp = row;
        colTmp = col;
        tileTR = TR;
        tileTC = TC;
    }
    TileWriteBack(dram, outFmOffset, cfgRowTmp, cfgColTmp, 
                    cfgM, rowTmp, colTmp, co, tileTR, tileTC);
}


/*
 * load input feature maps from (row*stride-boundary, col*stride-boundary) 
 * to ((row+Tr-1)*stride+boundary, (col+Tc-1)*stride+boundary) 
 * if the input feature maps points exceed the actual size as the pad operation,
 * the zero should be filled into the buffer.
 */
void InputTileLoad(DataType *dram, bool isPadding, ParameterType inFmOffset, ParameterType cfgN,
                    ParameterType cfgK, ParameterType cfgS, ParameterType cfgRow, 
                    ParameterType cfgCol, int row, int col, int ci) {
#pragma HLS inline off 
#pragma HLS resource variable=inputFmBuffer core=RAM_2P_BRAM
#pragma HLS array_partition variable=inputFmBuffer dim=1 complete
    int kernelBoundary = (cfgK-1)/2;
    ParameterType trrInMax = (row+TR-1)*cfgS + (isPadding ? 1: cfgK);
    ParameterType tccInMax = (col+TC-1)*cfgS + (isPadding ? 1: cfgK);
    ParameterType rowInMax = (cfgRow-1)*cfgS + (isPadding ? 1: cfgK);
    ParameterType colInMax = (cfgCol-1)*cfgS + (isPadding ? 1: cfgK);
    for(int tii = 0; tii < TN; ++tii) {
        for(int trr = 0; trr < TR_IN; ++trr) {
            for(int tcc = 0; tcc < TC_IN; ++tcc) {
#pragma HLS pipeline II=1
                int nPosition = ci + tii;
                int yPosition = (int)(trr+row*cfgS) + (isPadding ? (-kernelBoundary) : 0);
                int xPosition = (int)(tcc+col*cfgS) + (isPadding ? (-kernelBoundary) : 0);
                if(yPosition < trrInMax && xPosition < tccInMax) {
                    if(nPosition >= cfgN || yPosition < 0 || xPosition < 0 ||
                            yPosition >= rowInMax || xPosition >= colInMax)
                        inputFmBuffer[tii][trr][tcc] = 0;
                    else {
                        ParameterType immOffset = inFmOffset + nPosition*rowInMax*colInMax
                                                    + yPosition*colInMax + xPosition;
                        DataType readReg = *(dram + immOffset);
                        inputFmBuffer[tii][trr][tcc] = readReg;
                        //cout << "INPUT:" << nPosition << "\t" << yPosition << "\t" << xPosition <<
                        //    "\t" << "ADDR:" << (immOffset-inFmOffset) << "\t" << readReg << endl;
                    }
                }
            }
        }
    }
}
         
void WeightTileLoad(DataType *dram, ParameterType weightOffset, ParameterType cfgK, 
                    ParameterType cfgN, ParameterType cfgM, int co, 
                    int ci) {
#pragma HLS inline off
#pragma HLS resource variable=weightBuffer core=RAM_2P_BRAM
#pragma HLS array_partition variable=weightBuffer dim=1 complete
#pragma HLS array_partition variable=weightBuffer dim=2 complete
    for(int too = 0; too < MIN(TM,cfgM-co); ++too) {
        for(int tii = 0; tii < MIN(TN,cfgN-ci); ++tii) {
            for(int i = 0; i < K_MAX*K_MAX; ++i) {
#pragma HLS pipeline
                if(i < cfgK*cfgK) {
                    ParameterType immOffset = weightOffset + (too+co)*cfgN*cfgK*cfgK +
                                            (tii+ci)*cfgK*cfgK + i;
                    DataType readReg = *(dram + immOffset);
                    weightBuffer[too][tii][i] = readReg;
                }
            }
        }
    }
}

void BiasTileLoad(DataType *dram, ParameterType biasOffset, bool isFullConnect, 
                        ParameterType cfgCol, ParameterType cfgM, int col, int co) {
#pragma HLS inline off
#pragma HLS array_partition variable=biasBuffer dim=1 complete
    if(isFullConnect == 0) {
        for(int too = 0; too < TM; ++too) {
#pragma HLS pipeline
            if(too + co < cfgM)
                biasBuffer[too] = *(dram + biasOffset + co + too);
        }
    } else {
        for(int tcc = 0; tcc < TC; ++tcc) {
#pragma HLS pipeline
            if(tcc + col < cfgCol)
                biasBuffer[tcc] = *(dram + biasOffset + col + tcc);
        }
    }

}

void TileProcessEngine(DataType *dram, DataType *dram2, DataType *dram3, ParameterType weightOffset,
                        ParameterType biasOffset, ParameterType inFmOffset, bool isPadding,
                        bool isRelu, bool isFullConnect, ParameterType cfgRow, ParameterType cfgCol,
                        ParameterType cfgM, ParameterType cfgN, ParameterType cfgK, ParameterType cfgS,
                        int row, int col, int co, int ci) {
#pragma HLS inline off
#pragma HLS dataflow
    WeightTileLoad(dram, weightOffset, cfgK, cfgN, cfgM, co, ci);
    BiasTileLoad(dram2, biasOffset, isFullConnect, cfgCol, cfgM, col, co);
    InputTileLoad(dram3, isPadding, inFmOffset, cfgN, cfgK, cfgS, cfgRow, cfgCol, row, col, ci);
    TileConv(isRelu, isFullConnect, cfgRow,  cfgN, cfgK, cfgS, ci);

}

void TileProcessEngineWrapper(DataType *dram, DataType *dram2, DataType *dram3, ParameterType weightOffset,
                        ParameterType biasOffset, ParameterType inFmOffset, bool isPadding,
                        bool isRelu, bool isFullConnect, ParameterType cfgRow, ParameterType cfgCol,
                        ParameterType cfgM, ParameterType cfgN, ParameterType cfgK, ParameterType cfgS,
                        int row, int col, int co) {
#pragma HLS inline off
    for(int ci = 0; ci < cfgN; ci += TN) {
        TileProcessEngine(dram, dram2, dram3, weightOffset, biasOffset, inFmOffset, isPadding,
                        isRelu, isFullConnect, cfgRow, cfgCol, cfgM, cfgN, cfgK, cfgS, row, col, co, ci);
    }
}


void TileTop(DataType *dram, DataType *dram2, DataType *dram3, DataType *dram4,
                ParameterType weightOffset, ParameterType biasOffset, ParameterType inFmOffset,
                ParameterType outFmOffset, ParameterType isPadding, ParameterType isRelu,
                ParameterType isPoolingMax, ParameterType isFullConnect, ParameterType cfgRow,
                ParameterType cfgCol, ParameterType cfgM, ParameterType cfgN, ParameterType cfgK,
                ParameterType cfgS, ParameterType cfgPoolK, ParameterType cfgPoolS, 
                int row, int col, int co) {
#pragma HLS inline off
#pragma HLS dataflow 
    TileProcessEngineWrapper(dram, dram2, dram3, weightOffset, biasOffset, inFmOffset, isPadding,
                        isRelu, isFullConnect, cfgRow, cfgCol, cfgM, cfgN, cfgK, cfgS, row,
                        col, co);

    TilePooling(cfgN, cfgK, cfgPoolK, cfgPoolS);

    
    TileWriteBackWrapper(dram4, outFmOffset, isPoolingMax, cfgRow, cfgCol, 
                    cfgM, cfgPoolK, cfgPoolS, row, col, co);
}

void LayerTop(DataType *dram, DataType *dram2, DataType *dram3, DataType *dram4, LayerCfgType cfgSet) {
#pragma HLS interface m_axi port=dram offset=slave depth=436050 bundle=BUS_DATA1
#pragma HLS interface m_axi port=dram2 offset=slave depth=436050 bundle=BUS_DATA2
#pragma HLS interface m_axi port=dram3 offset=slave depth=436050 bundle=BUS_DATA3
#pragma HLS interface m_axi port=dram4 offset=slave depth=436050 bundle=BUS_DATA4
#pragma HLS interface s_axilite port=return
#pragma HLS interface s_axilite port=cfgSet


ParameterType weightOffset, biasOffset, inFmOffset, outFmOffset;
ParameterType cfgRow, cfgCol, cfgM, cfgN, cfgK, cfgS;
bool isRelu, isPoolingMax, isPadding, isFullConnect;
ParameterType cfgPoolS, cfgPoolK;

    weightOffset = cfgSet.weightOffset;
    biasOffset = cfgSet.biasOffset;
    inFmOffset = cfgSet.inFmOffset;
    outFmOffset = cfgSet.outFmOffset;
    isPadding = cfgSet.isPadding;
    isRelu = cfgSet.isRelu;
    isPoolingMax = cfgSet.isPoolingMax;
    isFullConnect = cfgSet.isFullConnect;
    cfgRow = cfgSet.cfgRow;
    cfgCol = cfgSet.cfgCol;
    cfgM = cfgSet.cfgM;
    cfgN = cfgSet.cfgN;
    cfgK = cfgSet.cfgK;
    cfgS = cfgSet.cfgS;
    cfgPoolK = cfgSet.cfgPoolK;
    cfgPoolS = cfgSet.cfgPoolS;
    
    assert(cfgRow < 2048);
    assert(cfgCol < 2048);
    assert(cfgM < 2048);
    assert(cfgN < 2048);
    assert(cfgK < K_MAX+1);
    assert(cfgS < S_MAX+1);
    assert(cfgPoolK < K_POOL_MAX+1);
    assert(cfgPoolS < S_POOL_MAX+1);


    for(int row = 0; row < cfgRow; row += TR) {
        for(int col = 0; col < cfgCol; col += TC) {
            for(int co = 0; co < cfgM; co += TM) {
                TileTop(dram, dram2, dram3, dram4, weightOffset, biasOffset, inFmOffset,
                        outFmOffset, isPadding, isRelu, isPoolingMax, isFullConnect, cfgRow,
                        cfgCol, cfgM, cfgN, cfgK, cfgS, cfgPoolK, cfgPoolS, row, col, co);
            }
        }
    }
}







