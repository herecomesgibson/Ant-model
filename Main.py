

import numpy as np

import pandas as pd

import seaborn as sns

import math

import matplotlib

import matplotlib.pyplot as plt

import random

import statistics as st

import AntSimulation as ans

#Parameter order 
#grid_size, num_cols, max_oc, ants_per_colony, alpha, beta, w, evap, pher_trail

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


paramlst = [ 20, 30, 40, 50 ] 
paramlstt = [ 2, 3, 4, 5, 6 ]


resultslst = []

for m in paramlst:
    resultslstt = []
    for k in paramlstt:
        
        vals = []
        
        for pe in range(10):
            ModelObj = ans.init_model(m, k, max_occ, ants_per_col, alpha, beta, w, evap, pher_trail)
            delivery_total = 0
            for ex in range(50):
                delivery_total += ans.update_model(ModelObj)
            
            vals.append(delivery_total)

        
        score = sum(vals)/len(vals)
        resultslstt.append(score)

    resultslst.append(resultslstt)


print(resultslst)

resarr = pd.DataFrame( resultslst)

print(resarr)


p1 = sns.heatmap(resarr, cmap="YlGnBu", yticklabels=paramlst, xticklabels=paramlstt, annot=True, fmt=".1f")
p1.invert_yaxis()
plt.xlabel('Number of colonies')
plt.ylabel('Grid size')
plt.title('Total deliveries heatmap by Grid size and number of colonies')
plt.show()



