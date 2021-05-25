
import numpy as np

import pandas as pd

import seaborn as sns

import math

import matplotlib

import matplotlib.pyplot as plt

import random

import statistics as st




paramlst = [ 10, 30, 50, 70 ]

x = [[627.5, 982.8, 739.7, 679.6], [669.6, 795.7, 903.2, 873.3], [723.5, 848.9, 857.1, 541.9]]

resarr = pd.DataFrame( x, columns=paramlst )

sns.heatmap(resarr, cmap="YlGnBu", yticklabels=[ 1, 3, 5 ], annot=True, fmt=".1f", vmin=200, vmax=2000)

plt.show()

