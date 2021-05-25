

import numpy as np

import pandas as pd

import seaborn as sns

import math

import matplotlib

import matplotlib.pyplot as plt

import random

import statistics as st

import AntSimulation as ans


'''
paramlst = [ 10, 30, 50, 70 ]

x = [[627.5, 982.8, 739.7, 679.6], [669.6, 795.7, 903.2, 873.3], [723.5, 848.9, 857.1, 541.9]]

resarr = pd.DataFrame( x, columns=paramlst )

sns.heatmap(resarr, cmap="YlGnBu", yticklabels=[ 1, 3, 5 ])

plt.show()
'''


#parameter sweep result variables

alpha = 1
beta = 1
grid_size = 20
num_colonies = 3
max_occ = 20
ants_per_col = 50
w=.5
evap = .9
pher_trail = 2


paramlst = [ 1, 3, 5 ]
paramlstt = [ 10, 30, 50, 70 ]


resultslst = []

for m in paramlst:
    resultslstt = []
    for k in paramlstt:
        
        vals = []
        
        for pe in range(20):
            ModelObj = ans.init_model(grid_size, m, max_occ, k, alpha, beta, w, evap, pher_trail)
            delivery_total = 0
            for ex in range(100):
                delivery_total += ans.update_model(ModelObj)
            
            vals.append(delivery_total)

        
        score = sum(vals)/len(vals)
        resultslstt.append(score)

    resultslst.append(resultslstt)




print(resultslst)

resarr = pd.DataFrame( resultslst)

print(resarr)


p1 = sns.heatmap(resarr, cmap="YlGnBu", yticklabels=paramlst, xticklabels=paramlstt)

plt.xlabel('Ants Per colony')
plt.ylabel('number of colonies')
plt.title('Total deliveries heatmap by colony count and ants per colony')
plt.show()



