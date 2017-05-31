#include <iostream>
#include "ip.hpp"

using namespace std;


static DataType biasBuffer[TM];
static DataType weightBuffer[TM][TN][K_MAX*K_MAX];
static DataType inputFmBuffer[TN][TR_IN][TC_IN];
static DataType immRegArray[TM];
static DataType immBuffer[TM][TR][TN];
static DataType immPoolArray[TM];
static DataType immPoolBuffer[TM][TR_POOL][TC_POOL];

void PE(ParameterType isRelu, ParameterType cfgK, ParameterType kernelPos,
            ParameterType trrInPos, ParameterType tccInPos, 
            ParameterType trrOutPos, ParameterType tccOutPos) {
#pragma HLS inline

    for(int too = 0; too < TM; ++too) {
#pragma HLS unroll
        DataType immReg = biasBuffer[too];
        for(int tii = 0; tii < TN; ++tii) {
#pragma HLS unroll
            immReg += weightBuffer[too][tii][kernelPos] *
                        inputFmBuffer[tii][trrInPos][tccInPos];
        }
        immRegArray[too] = immReg;
    }
    for(int too = 0; too < TM; ++too) {
#pragma HLS unroll
        DataType resultAdd = immBuffer[too][trrOutPos][tccOutPos];
        if(kernelPos == 0)
            immBuffer[too][trrOutPos][tccOutPos] = immRegArray[too];
        else if(kernelPos == cfgK*cfgK-1 && isRelu == 1)
            immBuffer[too][trrOutPos][tccOutPos] = resultAdd > 0 ? resultAdd : 0;
        else 
            immBuffer[too][trrOutPos][tccOutPos] = resultAdd;
    }

}

void TileConv(ParameterType isRelu, ParameterType cfgK, ParameterType cfgS) {
#pragma HLS inline
    for(int i = 0; i < cfgK; ++i) {
        for(int j = 0; j < cfgK; ++j) {
            ParameterType kernelPos = i*cfgK + j;
            for(int trr = 0; trr < TR; ++trr) {
                ParameterType trrInPos = trr * cfgS + i;
                for(int tcc = 0; tcc < TC; ++tcc) {
                    ParameterType tccInPos = tcc * cfgS + j;
                    PE(isRelu, cfgK, kernelPos, trrInPos, tccInPos, trr, tcc);
                }
            }
        }
    }
}

void TilePooling(ParameterType isPoolingMax, 
                    ParameterType poolK, ParameterType poolS) {
#pragma HLS inline
    if(isPoolingMax == 0)
        return;
    ParameterType trrOut = (TR-poolK)/poolS+1;
    ParameterType tccOut = (TC-poolK)/poolS+1;
    for(int trr = 0; trr < trrOut; ++trr) {
        for(int tcc = 0; tcc < tccOut; ++tcc) {
            for(int i = 0; i < poolK; ++i) {
                for(int j = 0; j < poolK; ++j) {
                    for(int too = 0; too < TM; ++too) {
#pragma HLS unroll
                        if(i == 0 && j == 0)
                            immPoolArray[too] = immBuffer[too][trr*poolS+i][tcc*poolS+j];
                        else
                            immPoolArray[too] = immBuffer[too][trr*poolS+i][tcc*poolS+j]
                                                > immPoolArray[too] ?
                                                immBuffer[too][trr*poolS+i][tcc*poolS+j]
                                                : immPoolArray[too];
                    }
                }
            }
            for(int too = 0; too < TM; ++too) {
#pragma HLS unroll
                immPoolBuffer[too][trr][tcc] = immPoolArray[too];
            }
        }
    }
}



void TileWriteBack(DataType *dram, ParameterType outFmOffset, ParameterType cfgRow,
        ParameterType cfgCol, ParameterType cfgM, ParameterType row, ParameterType col,
        ParameterType co) {
#pragma HLS inline
    for(int too = 0; too < MIN(TM, cfgM-co); ++too) {
        for(int trr = 0; trr < MIN(TR, cfgRow-row); ++ trr) {
            for(int tcc = 0; tcc < MIN(TC, cfgCol-col); ++ tcc) {
                DataType readReg = immBuffer[too][trr][tcc];
                ParameterType immOffset = outFmOffset + (too+co)*cfgRow*cfgCol
                                            + (trr+row)*cfgCol + col + tcc;
                *(dram + immOffset) = readReg;
            }
        }
    }
}

//load input feature maps from (row*stride-boundary, col*stride-boundary)
//to ((row+Tr-1)*stride+boundary, (col+Tc-1)*stride+boundary)
//if the input feature maps points exceed the actual size as the pad operation,
//the zero should be filled into the buffer.
void InputTileLoad(DataType *dram, ParameterType isPadding, ParameterType inFmOffset, ParameterType cfgN,
                    ParameterType cfgK, ParameterType cfgS, ParameterType cfgRow, 
                    ParameterType cfgCol, ParameterType row, ParameterType col,
                    ParameterType ci) {
#pragma HLS inline
    int kernelBoundary = (cfgK-1)/2;
    ParameterType trrInMax = (cfgRow-1)*cfgS + (isPadding ? 1: cfgK);
    ParameterType tccInMax = (cfgCol-1)*cfgS + (isPadding ? 1: cfgK);
    for(int tii = 0; tii < TN; ++tii) {
        for(int trr = 0; trr < TR_IN; ++trr) {
            for(int tcc = 0; tcc < TC_IN; ++tcc) {
#pragma HLS pipeline II=1
                cout << "InputTileLoad:" << tii << trr << tcc << endl;
                int yPosition = trr+row*cfgS + (isPadding ? (-kernelBoundary) : 0);
                int xPosition = tcc+col*cfgS + (isPadding ? (-kernelBoundary) : 0);
                if(yPosition < 0 || xPosition < 0 || yPosition > trrInMax || xPosition > tccInMax)
                    inputFmBuffer[tii][trr][tcc] = 0;
                else {
                    ParameterType immOffset = inFmOffset + (ci+tii)*trrInMax*tccInMax
                                        + yPosition*tccInMax + xPosition;
                    DataType readReg = *(dram + immOffset);
                    inputFmBuffer[tii][trr][tcc] = readReg;
                }
            }
        }
    }
}
         
void WeightTileLoad(DataType *dram, ParameterType weightOffset, ParameterType cfgK, 
                    ParameterType cfgN, ParameterType cfgM, ParameterType co, 
                    ParameterType ci) {
#pragma HLS inline 
    for(int too = 0; too < MIN(TM, cfgM-co); ++too) {
        for(int tii = 0; tii < MIN(TN, cfgN-ci); ++tii) {
            for(int i = 0; i < K_MAX*K_MAX; ++i) {
#pragma HLS pipeline II=1                
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

void LayerTop(DataType *dram, LayerCfgType cfgSet) {
ParameterType weightOffset, inFmOffset, outFmOffset;
ParameterType cfgRow, cfgCol, cfgM, cfgN, cfgK, cfgS;
ParameterType isRelu, isPoolingMax, isPadding;
ParameterType cfgPoolS, cfgPoolK;

    weightOffset = cfgSet.weightOffset;
    inFmOffset = cfgSet.inFmOffset;
    outFmOffset = cfgSet.outFmOffset;
    isPadding = cfgSet.isPadding;
    isRelu = cfgSet.isRelu;
    isPoolingMax = cfgSet.isPoolingMax;
    cfgRow = cfgSet.cfgRow;
    cfgCol = cfgSet.cfgCol;
    cfgM = cfgSet.cfgM;
    cfgN = cfgSet.cfgN;
    cfgK = cfgSet.cfgK;
    cfgS = cfgSet.cfgS;
    cfgPoolK = cfgSet.cfgPoolK;
    cfgPoolS = cfgSet.cfgPoolS;


    for(int row = 0; row < cfgRow; row += TR) {
        for(int col = 0; col < cfgCol; col += TC) {
            for(int co = 0; co < cfgM; co += TM) {
                for(int ci = 0; ci < cfgN; ci += TN) {
                    cout << row << col << co << ci << endl;
                    WeightTileLoad(dram, weightOffset, cfgK, cfgN, cfgM, co, ci);
                    InputTileLoad(dram, isPadding, inFmOffset, cfgN, cfgK, cfgS, cfgRow, cfgCol,
                                    row, col, ci);
                    TileConv(isRelu, cfgK, cfgS);
                    TilePooling(isPoolingMax, cfgPoolK, cfgPoolS);
                    TileWriteBack(dram, outFmOffset, cfgRow, cfgCol, cfgM, row,
                                        col, co);
                }
            }
        }
    }
}








                                            


                










