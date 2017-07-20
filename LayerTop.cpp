#include <iostream>
#include <assert.h>
#include "LayerTop.hpp"

using namespace std;




void elementUnit(bool isRelu, bool isFullConnect, ParameterType cfgN, ParameterType cfgK,
                    int ci, int trr, int tcc, int kernelPos, int trrInPos, int tccInPos, bool sel,
                    DataType biasBuffer[BIAS_NUM], DataType weightBuffer[TM][TN][K_MAX*K_MAX],
                    DataType inputFmBuffer[TN][TR_IN][TC_IN], DataType immBuffer2[TM][TR][TC]) {
#pragma HLS inline off
static DataType immBuffer[2][TM][TR][TC];
#pragma HLS resource variable=immBuffer core=RAM_2P_BRAM
#pragma HLS array_partition variable=immBuffer dim=1 complete
#pragma HLS array_partition variable=immBuffer dim=2 complete
#pragma HLS dependence variable=immBuffer array inter false

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
        DataType tmp, tmp2;
        if(ci == 0 && kernelPos == 0) 
        	tmp = biasBuffer[biasBufferIndex] + immReg;
        else if(ci+TN >= cfgN && kernelPos == cfgK*cfgK-1) {
            DataType localReg = isRelu ? (resultAdd > 0 ? resultAdd : 0) : resultAdd;
            tmp2 = localReg;
        } else
        	tmp = resultAdd;

        immBuffer[sel][too][trr][tcc] = tmp;
        immBuffer2[too][trr][tcc] = tmp2;
    }
}


void PE(bool isRelu, bool isFullConnect, ParameterType cfgRow, ParameterType cfgN, ParameterType cfgK,
            ParameterType cfgS, int ci, int i, int j, bool sel, DataType biasBuffer[BIAS_NUM],
            DataType weightBuffer[TM][TN][K_MAX*K_MAX], DataType inputFmBuffer[TN][TR_IN][TC_IN],
            DataType immBuffer2[TM][TR][TC]
            ) {
#pragma HLS inline off
    for(int trr = 0; trr < cfgRow; ++trr) {
        for(int tcc = 0; tcc < TC; ++tcc) {
#pragma HLS pipeline
            int kernelPos = i*cfgK + j;
            int trrInPos = trr * cfgS + i;
            int tccInPos = tcc * cfgS + j;

            elementUnit(isRelu, isFullConnect, cfgN, cfgK, ci, trr, tcc, kernelPos, trrInPos, tccInPos, sel,
                            biasBuffer, weightBuffer, inputFmBuffer, immBuffer2);

        }
    }
}


void TileConv(bool isRelu, bool isFullConnect, ParameterType cfgRow, ParameterType cfgN, ParameterType cfgK,
                ParameterType cfgS, int ci, DataType biasBuffer[BIAS_NUM], DataType weightBuffer[TM][TN][K_MAX*K_MAX], 
                DataType inputFmBuffer[TN][TR_IN][TC_IN], DataType immBuffer2[TM][TR][TC]) {
#pragma HLS inline off

    static bool sel = 0;
    for(int i = 0; i < cfgK; ++i) {
        for(int j = 0; j < cfgK; ++j) {
            if(ci == 0 && i == 0 && j == 0)
                sel = 0;
            else 
                sel = !sel;
            	PE(isRelu, isFullConnect, cfgRow, cfgN, cfgK, cfgS, ci, i, j, sel, biasBuffer, weightBuffer,
                        inputFmBuffer, immBuffer2);
        }
    }
}

void TilePooling(ParameterType cfgN, ParameterType cfgK, ParameterType poolK, ParameterType poolS,
                    DataType immBuffer2[TM][TR][TC], DataType immPoolBuffer[TM][TR][TC]) {
#pragma HLS inline off
static DataType immPoolArray[TM];
#pragma HLS array_partition variable=immPoolArray dim=1 complete
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
                            DataType tmpReg = immBuffer2[too][trr*poolS+i][tcc*poolS+j];
                            if(i == 0 && j == 0)
                                immPoolArray[too] = tmpReg;
                            else
                                immPoolArray[too] = tmpReg > immPoolArray[too] ? tmpReg : immPoolArray[too];
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
        ParameterType co, ParameterType tileTR, ParameterType tileTC, DataType immPoolBuffer[TM][TR][TC]) {
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
                    //cout << immOffset << endl;
                }
                //if(too == 0)
                //    cout << "WB:" << too << "\t" << trr << "\t" << tcc << "\t" << (immOffset-outFmOffset+1) << ": " << readReg << endl;
            }
        }
    }
}

void TileWriteBackWrapper(DataType *dram, ParameterType outFmOffset, bool isPoolingMax, ParameterType cfgRow,
        ParameterType cfgCol, ParameterType cfgM, ParameterType cfgPoolK, ParameterType cfgPoolS, 
        ParameterType row, ParameterType col, ParameterType co, DataType immPoolBuffer[TM][TR][TC]) {
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
                    cfgM, rowTmp, colTmp, co, tileTR, tileTC, immPoolBuffer);
}


/*
 * load input feature maps from (row*stride-boundary, col*stride-boundary) 
 * to ((row+Tr-1)*stride+boundary, (col+Tc-1)*stride+boundary) 
 * if the input feature maps points exceed the actual size as the pad operation,
 * the zero should be filled into the buffer.
 */
void InputTileLoad(DataType *dram, bool isPadding, ParameterType inFmOffset, ParameterType cfgN,
        ParameterType cfgK, ParameterType cfgS, ParameterType cfgRow, 
        ParameterType cfgCol, int row, int col, int ci, DataType inputFmBuffer[TN][TR_IN][TC_IN]) {
#pragma HLS inline off 

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
         
void WeightTileLoad(DataType *dram, ParameterType weightOffset, ParameterType biasOffset, bool isFullConnect,
                        ParameterType cfgCol, ParameterType cfgK, ParameterType cfgN, ParameterType cfgM, 
                        int col, int co, int ci, DataType weightBuffer[TM][TN][K_MAX*K_MAX], 
                        DataType biasBuffer[BIAS_NUM]) {
#pragma HLS inline off

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





void TileProcessEngine(DataType *dram, DataType *dram2, ParameterType weightOffset,
                        ParameterType biasOffset, ParameterType inFmOffset, bool isPadding,
                        bool isRelu, bool isFullConnect, ParameterType cfgRow, ParameterType cfgCol,
                        ParameterType cfgM, ParameterType cfgN, ParameterType cfgK, ParameterType cfgS,
                        int row, int col, int co, int ci, DataType immBuffer2[TM][TR][TC]) {
#pragma HLS inline
    static DataType biasBuffer0[BIAS_NUM];
#pragma HLS array_partition variable=biasBuffer0 dim=1 complete
    static DataType biasBuffer1[BIAS_NUM];
#pragma HLS array_partition variable=biasBuffer1 dim=1 complete
    static DataType weightBuffer0[TM][TN][K_MAX*K_MAX];
#pragma HLS resource variable=weightBuffer0 core=RAM_2P_BRAM
#pragma HLS array_partition variable=weightBuffer0 dim=1 complete
#pragma HLS array_partition variable=weightBuffer0 dim=2 complete
    static DataType weightBuffer1[TM][TN][K_MAX*K_MAX];
#pragma HLS resource variable=weightBuffer1 core=RAM_2P_BRAM
#pragma HLS array_partition variable=weightBuffer1 dim=1 complete
#pragma HLS array_partition variable=weightBuffer1 dim=2 complete
    static DataType inputFmBuffer0[TN][TR_IN][TC_IN];
#pragma HLS resource variable=inputFmBuffer0 core=RAM_2P_BRAM
#pragma HLS array_partition variable=inputFmBuffer0 dim=1 complete
    static DataType inputFmBuffer1[TN][TR_IN][TC_IN];
#pragma HLS resource variable=inputFmBuffer1 core=RAM_2P_BRAM
#pragma HLS array_partition variable=inputFmBuffer1 dim=1 complete

    static enum {TPE_START = 0, TPE_RUN = 1} tpeState = TPE_START;


    static bool sel = 0;
    sel = ci == 0 ? 0 : !sel;

    static int ci2 = 0;


    switch(tpeState) {
        case TPE_START:
            WeightTileLoad(dram, weightOffset, biasOffset, isFullConnect, cfgCol, cfgK, cfgN, cfgM, col, co, ci, weightBuffer0, biasBuffer0);
            InputTileLoad(dram2, isPadding, inFmOffset, cfgN, cfgK, cfgS, cfgRow, cfgCol, row, col, ci, inputFmBuffer0);
            if(ci+1+TN > cfgN) {
                TileConv(isRelu, isFullConnect, cfgRow,  cfgN, cfgK, cfgS, ci, biasBuffer0, weightBuffer0, inputFmBuffer0, immBuffer2);
                tpeState = TPE_START;
            }
            else
                tpeState = TPE_RUN;
            break;

        case TPE_RUN:
            if(sel == 0) {
                TileConv(isRelu, isFullConnect, cfgRow,  cfgN, cfgK, cfgS, ci2, biasBuffer1, weightBuffer1, inputFmBuffer1, immBuffer2);
                WeightTileLoad(dram, weightOffset, biasOffset, isFullConnect, cfgCol, cfgK, cfgN, cfgM, col, co, ci, weightBuffer0, biasBuffer0);
                InputTileLoad(dram2, isPadding, inFmOffset, cfgN, cfgK, cfgS, cfgRow, cfgCol, row, col, ci, inputFmBuffer0);
            } else {
                TileConv(isRelu, isFullConnect, cfgRow,  cfgN, cfgK, cfgS, ci2, biasBuffer0, weightBuffer0, inputFmBuffer0, immBuffer2);
                WeightTileLoad(dram, weightOffset, biasOffset, isFullConnect, cfgCol, cfgK, cfgN, cfgM, col, co, ci, weightBuffer1, biasBuffer1);
                InputTileLoad(dram2, isPadding, inFmOffset, cfgN, cfgK, cfgS, cfgRow, cfgCol, row, col, ci, inputFmBuffer1);
            }

            if(ci+1+TN > cfgN) {
                if(sel == 0)
                    TileConv(isRelu, isFullConnect, cfgRow,  cfgN, cfgK, cfgS, ci, biasBuffer0, weightBuffer0, inputFmBuffer0, immBuffer2);
                else
                    TileConv(isRelu, isFullConnect, cfgRow,  cfgN, cfgK, cfgS, ci, biasBuffer1, weightBuffer1, inputFmBuffer1, immBuffer2);
                tpeState = TPE_START;
            }
            break;
    }
    

    ci2 = ci;
    
    //WeightTileLoad(dram, weightOffset, biasOffset, isFullConnect, cfgCol, cfgK, cfgN, cfgM, col, co, ci, weightBuffer, biasBuffer);
    //InputTileLoad(dram3, isPadding, inFmOffset, cfgN, cfgK, cfgS, cfgRow, cfgCol, row, col, ci, inputFmBuffer);
    //TileConv(isRelu, isFullConnect, cfgRow,  cfgN, cfgK, cfgS, ci, biasBuffer, weightBuffer, inputFmBuffer);

}

void TileProcessEngineWrapper(DataType *dram, DataType *dram2, ParameterType weightOffset,
                        ParameterType biasOffset, ParameterType inFmOffset, bool isPadding,
                        bool isRelu, bool isFullConnect, ParameterType cfgRow, ParameterType cfgCol,
                        ParameterType cfgM, ParameterType cfgN, ParameterType cfgK, ParameterType cfgS,
                        int row, int col, int co, DataType immBuffer2[TM][TR][TC]) {
#pragma HLS inline off
    for(int ci = 0; ci < cfgN; ci += TN) {
    	TileProcessEngine(dram, dram2, weightOffset, biasOffset, inFmOffset, isPadding,
                            isRelu, isFullConnect, cfgRow, cfgCol, cfgM, cfgN, cfgK, cfgS, row,
                            col, co, ci, immBuffer2);
    }
}


void TileTop(DataType *dram, DataType *dram2, DataType *dram3,
                ParameterType weightOffset, ParameterType biasOffset, ParameterType inFmOffset,
                ParameterType outFmOffset, ParameterType isPadding, ParameterType isRelu,
                ParameterType isPoolingMax, ParameterType isFullConnect, ParameterType cfgRow,
                ParameterType cfgCol, ParameterType cfgM, ParameterType cfgN, ParameterType cfgK,
                ParameterType cfgS, ParameterType cfgPoolK, ParameterType cfgPoolS, 
                int row, int col, int co) {
#pragma HLS inline off
static DataType immBuffer2[TM][TR][TC];
#pragma HLS resource variable=immBuffer2 core=RAM_2P_BRAM  
#pragma HLS array_partition variable=immBuffer2 dim=1 complete
static DataType immPoolBuffer0[TM][TR][TC];
#pragma HLS resource variable=immPoolBuffer0 core=RAM_2P_BRAM
#pragma HLS array_partition variable=immPoolBuffer0 dim=1 complete
static DataType immPoolBuffer1[TM][TR][TC];
#pragma HLS resource variable=immPoolBuffer1 core=RAM_2P_BRAM
#pragma HLS array_partition variable=immPoolBuffer1 dim=1 complete


    static enum {TT_START = 0, TT_RUN = 1} ttState = TT_START;

    static bool sel = 0;
    if(row == 0 && col == 0 && co == 0)
        sel = 0;
    else 
        sel = !sel;

    static int row2=0, col2=0, co2=0;
    bool last = (row+1+TR > cfgRow) && (col+1+TC > cfgCol) && (co+1+TM > cfgM);

    switch(ttState) {
        case TT_START:
            TileProcessEngineWrapper(dram, dram2, weightOffset, biasOffset, inFmOffset, isPadding,
                        isRelu, isFullConnect, cfgRow, cfgCol, cfgM, cfgN, cfgK, cfgS, row,
                        col, co, immBuffer2);

            TilePooling(cfgN, cfgK, cfgPoolK, cfgPoolS, immBuffer2, immPoolBuffer0);
            if(last) {
                TileWriteBackWrapper(dram3, outFmOffset, isPoolingMax, cfgRow, cfgCol, 
                        cfgM, cfgPoolK, cfgPoolS, row, col, co, immPoolBuffer0);
                ttState = TT_START;
            } else
                ttState = TT_RUN;
            break;

        case TT_RUN: 
            if(sel == 0) {
                TileWriteBackWrapper(dram3, outFmOffset, isPoolingMax, cfgRow, cfgCol, 
                        cfgM, cfgPoolK, cfgPoolS, row2, col2, co2, immPoolBuffer1);
                TileProcessEngineWrapper(dram, dram2, weightOffset, biasOffset, inFmOffset, isPadding,
                            isRelu, isFullConnect, cfgRow, cfgCol, cfgM, cfgN, cfgK, cfgS, row,
                            col, co, immBuffer2);
                TilePooling(cfgN, cfgK, cfgPoolK, cfgPoolS, immBuffer2, immPoolBuffer0);
            } else {
                TileWriteBackWrapper(dram3, outFmOffset, isPoolingMax, cfgRow, cfgCol, 
                        cfgM, cfgPoolK, cfgPoolS, row2, col2, co2, immPoolBuffer0);
                TileProcessEngineWrapper(dram, dram2, weightOffset, biasOffset, inFmOffset, isPadding,
                            isRelu, isFullConnect, cfgRow, cfgCol, cfgM, cfgN, cfgK, cfgS, row,
                            col, co, immBuffer2);
                TilePooling(cfgN, cfgK, cfgPoolK, cfgPoolS, immBuffer2, immPoolBuffer1);
            }
            if(last) {
                if(sel == 0) 
                    TileWriteBackWrapper(dram3, outFmOffset, isPoolingMax, cfgRow, cfgCol, 
                        cfgM, cfgPoolK, cfgPoolS, row, col, co, immPoolBuffer0);
                else 
                    TileWriteBackWrapper(dram3, outFmOffset, isPoolingMax, cfgRow, cfgCol, 
                        cfgM, cfgPoolK, cfgPoolS, row, col, co, immPoolBuffer1);
                ttState = TT_START;
            }   
            break;
    }

    row2 = row;
    col2 = col;
    co2 = co;

/*
    TileProcessEngineWrapper(dram, dram2, weightOffset, biasOffset, inFmOffset, isPadding,
                        isRelu, isFullConnect, cfgRow, cfgCol, cfgM, cfgN, cfgK, cfgS, row,
                        col, co, immBuffer2);

    TilePooling(cfgN, cfgK, cfgPoolK, cfgPoolS, immBuffer2, immPoolBuffer);

    
    TileWriteBackWrapper(dram3, outFmOffset, isPoolingMax, cfgRow, cfgCol, 
                    cfgM, cfgPoolK, cfgPoolS, row, col, co, immPoolBuffer);
*/

}

void LayerTop(DataType *dram, DataType *dram2, DataType *dram3, const LayerCfgType cfgSet) {
#pragma HLS interface m_axi port=dram offset=slave depth=436054 bundle=BUS_DATA1
#pragma HLS interface m_axi port=dram2 offset=slave depth=436054 bundle=BUS_DATA2
#pragma HLS interface m_axi port=dram3 offset=slave depth=436054 bundle=BUS_DATA3
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
    
    assert(cfgRow < 1024);
    assert(cfgCol < 1024);
    assert(cfgM < 1024);
    assert(cfgN < 1024);
    assert(cfgK < K_MAX+1);
    assert(cfgS < S_MAX+1);
    assert(cfgPoolK < K_POOL_MAX+1);
    assert(cfgPoolS < S_POOL_MAX+1);


    for(int row = 0; row < cfgRow; row += TR) {
        for(int col = 0; col < cfgCol; col += TC) {
            for(int co = 0; co < cfgM; co += TM) {
                TileTop(dram, dram2, dram3, weightOffset, biasOffset, inFmOffset,
                        outFmOffset, isPadding, isRelu, isPoolingMax, isFullConnect, cfgRow,
                        cfgCol, cfgM, cfgN, cfgK, cfgS, cfgPoolK, cfgPoolS, row, col, co);
            }
        }
    }
}







