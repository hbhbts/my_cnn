#include <iostream>
#include <assert.h>
#include "LayerTop.hpp"

using namespace std;


static DataType biasBuffer[BIAS_NUM];
static DataType weightBuffer[TM][TN][K_MAX*K_MAX];
static DataType inputFmBuffer[TN][TR_IN][TC_IN];
static DataType immRegArray[TM];
static DataType immBuffer[TM][TR][TC];
static DataType immPoolArray[TM];
static DataType immPoolBuffer[TM][TR_POOL][TC_POOL];

void PE(bool isRelu, bool isFullConnect, ParameterType cfgN, ParameterType cfgK, ParameterType kernelPos,
            ParameterType trrInPos, ParameterType tccInPos, 
            ParameterType trrOutPos, ParameterType tccOutPos, ParameterType ci) {
#pragma HLS inline
#pragma HLS array_partition variable=immRegArray dim=1 complete
#pragma HLS array_partition variable=immBuffer dim=1 complete
#pragma HLS resource variable=immBuffer core=RAM_2P_BRAM

    for(int too = 0; too < TM; ++too) {
#pragma HLS unroll
        DataType immReg = 0;
        for(int tii = 0; tii < TN; ++tii) {
#pragma HLS unroll
            immReg += weightBuffer[too][tii][kernelPos] *
                        inputFmBuffer[tii][trrInPos][tccInPos];
            //if(too == 1 && trrOutPos == 0 && tccOutPos == 0)
            //    cout << "PEIN: " << weightBuffer[too][tii][kernelPos] << "\t" << inputFmBuffer[tii][trrInPos][tccInPos] << "\t" << endl;
            

        }
        immRegArray[too] = immReg;
    }

    for(int too = 0; too < TM; ++too) {
#pragma HLS unroll
        int biasBufferIndex = isFullConnect ? tccOutPos : too;
        DataType resultAdd = immBuffer[too][trrOutPos][tccOutPos] + immRegArray[too];
        if(ci == 0 && kernelPos == 0) 
            immBuffer[too][trrOutPos][tccOutPos] = biasBuffer[biasBufferIndex] + immRegArray[too];
        else if(ci+TN >= cfgN && kernelPos == cfgK*cfgK-1 && isRelu == true)
            immBuffer[too][trrOutPos][tccOutPos] = resultAdd > 0 ? resultAdd : 0;
        else
            immBuffer[too][trrOutPos][tccOutPos] = resultAdd;
        //if(too == 1 && trrOutPos == 0 && tccOutPos == 0)
        //    cout << "PE: " << trrOutPos << "\t" << tccOutPos << "\t" << immBuffer[too][trrOutPos][tccOutPos] << "\t" << biasBuffer[biasBufferIndex] << endl;

    }


}

void TileConv(bool isRelu, bool isFullConnect, ParameterType cfgN, ParameterType cfgK, ParameterType cfgS, ParameterType ci) {
#pragma HLS inline off
    for(int i = 0; i < cfgK; ++i) {
        for(int j = 0; j < cfgK; ++j) {
            ParameterType kernelPos = i*cfgK + j;
            for(int trr = 0; trr < TR; ++trr) {
                ParameterType trrInPos = trr * cfgS + i;
                for(int tcc = 0; tcc < TC; ++tcc) {
#pragma HLS pipeline 
                    ParameterType tccInPos = tcc * cfgS + j;
                    PE(isRelu, isFullConnect, cfgN, cfgK, kernelPos, trrInPos, tccInPos, trr, tcc, ci);

                }
            }
        }
    }
}

void TilePooling(ParameterType poolK, ParameterType poolS) {
#pragma HLS inline
#pragma HLS array_partition variable=immPoolArray dim=1 complete
#pragma HLS array_partition variable=immPoolBuffer dim=1 complete
#pragma HLS resource variable=immPoolBuffer core=RAM_2P_BRAM
    ParameterType trrOut = (TR-poolK)/poolS+1; //need to be integer
    ParameterType tccOut = (TC-poolK)/poolS+1; //need to be integer
    for(int trr = 0; trr < trrOut; ++trr) {
        for(int tcc = 0; tcc < tccOut; ++tcc) {
            for(int i = 0; i < poolK; ++i) {
                for(int j = 0; j < poolK; ++j) {
#pragma HLS pipeline
                    for(int too = 0; too < TM; ++too) {
#pragma HLS unroll
                        if(i == 0 && j == 0)
                            immPoolArray[too] = immBuffer[too][trr*poolS+i][tcc*poolS+j];
                        else
                            immPoolArray[too] = immBuffer[too][trr*poolS+i][tcc*poolS+j]
                                                > immPoolArray[too] ?
                                                immBuffer[too][trr*poolS+i][tcc*poolS+j]
                                                : immPoolArray[too];
                        //if(too == 0)
                        //    cout << "LOOP: " << immBuffer[too][trr*poolS+i][tcc*poolS+j] << endl;
                    }
                }
            }
            for(int too = 0; too < TM; ++too) {
#pragma HLS unroll
                immPoolBuffer[too][trr][tcc] = immPoolArray[too];
                //if(too == 0)
                //    cout << "POOL: " << too << "\t" << trr << "\t" << tcc << "\t" << immPoolArray[too] << endl;
            }
        }
    }
}



void TileWriteBack(DataType *dram, ParameterType outFmOffset, bool isPoolingMax, ParameterType cfgRow,
        ParameterType cfgCol, ParameterType cfgM, ParameterType row, ParameterType col,
        ParameterType co, ParameterType tileTR, ParameterType tileTC) {
#pragma HLS inline
    for(int too = 0; too < MIN(TM, cfgM-co); ++too) {
        for(int trr = 0; trr < MIN(tileTR, cfgRow-row); ++ trr) {
            for(int tcc = 0; tcc < MIN(tileTC, cfgCol-col); ++ tcc) {
                DataType readReg;
                if(isPoolingMax)
                    readReg = immPoolBuffer[too][trr][tcc];
                else 
                    readReg = immBuffer[too][trr][tcc];
                ParameterType immOffset = outFmOffset + (too+co)*cfgRow*cfgCol
                                            + (trr+row)*cfgCol + col + tcc;
                *(dram + immOffset) = readReg;
                //if(too == 0)
                //    cout << "WB:" << too << "\t" << trr << "\t" << tcc << "\t" << (immOffset-outFmOffset+1) << ": " << readReg << endl;
            }
        }
    }
}

/*
 * load input feature maps from (row*stride-boundary, col*stride-boundary) 
 * to ((row+Tr-1)*stride+boundary, (col+Tc-1)*stride+boundary) 
 * if the input feature maps points exceed the actual size as the pad operation,
 * the zero should be filled into the buffer.
 */
void InputTileLoad(DataType *dram, bool isPadding, ParameterType inFmOffset, ParameterType cfgN,
                    ParameterType cfgK, ParameterType cfgS, ParameterType cfgRow, 
                    ParameterType cfgCol, ParameterType row, ParameterType col,
                    ParameterType ci) {
#pragma HLS inline off 
#pragma HLS array_partition variable=inputFmBuffer dim=1 complete
#pragma HLS resource variable=inputFmBuffer core=RAM_2P_BRAM
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
                    ParameterType cfgN, ParameterType cfgM, ParameterType co, 
                    ParameterType ci) {
#pragma HLS inline off
#pragma HLS array_partition variable=weightBuffer dim=1 complete
#pragma HLS array_partition variable=weightBuffer dim=2 complete
#pragma HLS resource variable=weightBuffer core=RAM_2P_BRAM
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

void BiasTileLoad(DataType *dram, ParameterType biasOffset, bool isFullConnect, 
                        ParameterType cfgCol, ParameterType cfgM, ParameterType col, ParameterType co) {
#pragma HLS inline off
#pragma HLS array_partition variable=biasBuffer dim=1 complete
    if(isFullConnect == 0) {
        for(int too = 0; too < TM; ++too) {
            if(too + co < cfgM)
                biasBuffer[too] = *(dram + biasOffset + co + too);
        }
    } else {
        for(int tcc = 0; tcc < TC; ++tcc) {
            if(tcc + col < cfgCol)
                biasBuffer[tcc] = *(dram + biasOffset + col + tcc);
        }
    }

}

void LayerTop(DataType *dram, LayerCfgType cfgSet) {
#pragma HLS interface m_axi port=dram offset=slave depth=749416 bundle=BUS_DATA
#pragma HLS interface s_axilite port=return bundle=BUS_REG
#pragma HLS interface s_axilite port=cfgSet bundle=BUS_REG

assert(cfgSet.cfgRow < 2048);
assert(cfgSet.cfgCol < 2048);
assert(cfgSet.cfgM < 2048);
assert(cfgSet.cfgN < 2048);
assert(cfgSet.cfgK < K_MAX+1);
assert(cfgSet.cfgS < S_MAX+1);
assert(cfgSet.cfgPoolK < K_POOL_MAX+1);
assert(cfgSet.cfgPoolS < S_POOL_MAX+1);


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
    DataType *dram2, *dram3, *dram4;
    dram4 = dram3 = dram2 = dram;
    


    for(int row = 0; row < cfgRow; row += TR) {
        for(int col = 0; col < cfgCol; col += TC) {
            for(int co = 0; co < cfgM; co += TM) {
                int cfgRowTmp, cfgColTmp, rowTmp, colTmp, tileTR, tileTC;
                for(int ci = 0; ci < cfgN; ci += TN) {
#pragma HLS dataflow
                    //cout << "Top: " << row << "\t" << col << "\t" << co << "\t" << ci << "\t" << endl;
                    WeightTileLoad(dram, weightOffset, cfgK, cfgN, cfgM, co, ci);
                    BiasTileLoad(dram2, biasOffset, isFullConnect, cfgCol, cfgM, col, co);
                    InputTileLoad(dram3, isPadding, inFmOffset, cfgN, cfgK, cfgS, cfgRow, cfgCol,
                                    row, col, ci);
                    TileConv(isRelu, isFullConnect, cfgN, cfgK, cfgS, ci);
                }
                cfgRowTmp = cfgRow;
                cfgColTmp = cfgCol;
                rowTmp = row;
                colTmp = col;
                tileTR = TR;
                tileTC = TC;
                if(isPoolingMax) {
                    TilePooling(cfgPoolK, cfgPoolS);
                    cfgRowTmp = (cfgRow-cfgPoolK)/cfgPoolS + 1;
                    cfgColTmp = (cfgCol-cfgPoolK)/cfgPoolS + 1;
                    rowTmp = (row/cfgPoolK);
                    colTmp = (col/cfgPoolK);
                    tileTR = (TR-cfgPoolK)/cfgPoolS + 1;
                    tileTC = (TC-cfgPoolK)/cfgPoolS + 1;
                }
                TileWriteBack(dram4, outFmOffset, isPoolingMax, cfgRowTmp, cfgColTmp, 
                                cfgM, rowTmp, colTmp, co, tileTR, tileTC);
            }
        }
    }
}







