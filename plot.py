import matplotlib.pyplot as plt
import numpy as np
import math

F = 100 #MHz
N = 192
M = 128
R = 13
C = 13
K = 3
S = 1

ratio = 0.80

DSP = 2800 * ratio
LUT = 303600 * ratio
FF = 607200 * ratio
BRAM = 2060*1024 * ratio


value = []

for Tr in np.arange(1.0, R+1, 1):
    for Tc in np.arange(1.0, C+1, 1):
        for Tn in np.arange(1.0, N+1, 1):
            for Tm in np.arange(1.0, M+1, 1):
                if(Tm*(Tn*3+(Tn-1)*2) < DSP):
                    if(Tm*(Tn*135 + (Tn-1)*214) < LUT):
                        if(Tm*(Tn*128 + (Tn-1)*227) < FF):
                            if(Tn*(S*(Tr-1)+K)*(S*(Tc-1)+K)+Tm*Tn*K*K+Tm*Tr*Tc < BRAM):
                                value.append([Tr, Tc, Tm, Tn])


y = [((M*R*C*N*K*K*2*F/1000)/ \
        (math.ceil(M/tm)*math.ceil(N/tn)*(R/tr)*(C/tc)*(10*math.ceil(math.log(tn,2))+tr*tc*K*K)), \
        tr, tc, tm, tn)
        for tr, tc, tm, tn in value]


x = [((R*C*M*N*K*K*2/4)/ \
        ((R/tr*C/tc*M/tm*N/tn)*(tm*tn*K*K + tn*(S*(tc-1)+K)*(S*(tr-1)+K)) \
        + (R/tr*C/tc*M/tm)*(tm*tr*tc)), tr, tc, tm, tn) \
        for tr, tc, tm, tn in value]
xx = [x1[0] for x1 in x]
yy = [y1[0] for y1 in y]


for i in range(0, len(x)):
    if(x[i][0] > 55 and y[i][0] > 80):
        print("CTC=%f\tCP=%f\tTr=%f\tTc=%f\tTm=%f\tTn=%f\tBW=%f" % (x[i][0], y[i][0], y[i][1], y[i][2], y[i][3], y[i][4], y[i][0]/x[i][0]))

plt.plot(xx, yy, '.')

plt.show()



