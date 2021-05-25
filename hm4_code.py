

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
max_occ = 5
ants_per_col = 50
w=.5
evap = .9
pher_trail = 2


paramlst = [ .1, .2, .3, .4, .5, .6, .7, .8, .9 ] 
paramlstt = [ 1, 2, 3, 4, 5 ]


resultslst = []

for m in paramlst:
    resultslstt = []
    for k in paramlstt:
        
        vals = []
        
        for pe in range(5):
            ModelObj = ans.init_model(grid_size, k, max_occ, ants_per_col, alpha, beta, m, evap, pher_trail)
            delivery_total = 0
            for ex in range(100):
                delivery_total += ans.update_model(ModelObj)
            
            vals.append( (delivery_total / (5 * k)) )

        
        score = sum(vals)/len(vals)
        resultslstt.append(score)

    resultslst.append(resultslstt)


print(resultslst)

resarr = pd.DataFrame( resultslst)

print(resarr)
#, annot=True, fmt=".1f"

p1 = sns.heatmap(resarr, cmap="YlGnBu", yticklabels=paramlst, xticklabels=paramlstt)
p1.invert_yaxis()
plt.xlabel('Number of colonies')
plt.ylabel('w')
plt.title('Average deliveries per Colony')
plt.show()



