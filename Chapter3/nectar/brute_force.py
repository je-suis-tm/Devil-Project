# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 04:16:29 2025

@author: Administrator
"""

import numpy as np
from itertools import product
import pandas as pd

lhs=[]
lhs.append(np.arange(200,310,100))
lhs.append(np.arange(400,610,100))
lhs.append(np.arange(15,21,2))
lhs.append(np.arange(7,11,2))
lhs.append(np.arange(0.5,1,0.1))
lhs.append(np.arange(180,550,150))
lhs.append(np.arange(90,190,60))
lhs.append(np.arange(0.1,1,0.4))
lhs.append(np.arange(1,4))
lhs.append(np.arange(0.5,1,0.1))
lhs.append(np.arange(180,550,150))
lhs.append(np.arange(90,190,60))
lhs.append(np.arange(0.1,1,0.4))
lhs.append(np.arange(1,4))


# Use itertools.product to get the Cartesian product (all combinations)
combinations = list(product(*lhs))

# Convert the list of combinations to a numpy array
combinations_array = np.array(combinations)

df=pd.DataFrame(combinations_array,columns=range(len(lhs)))
df.to_csv('brute_force.csv',index=False)


lhs.append(np.arange(365,1096,365))
lhs.append(np.arange(365,1096,365))

# Use itertools.product to get the Cartesian product (all combinations)
combinations = list(product(*lhs))

# Convert the list of combinations to a numpy array
combinations_array = np.array(combinations)

df=pd.DataFrame(combinations_array,columns=range(len(lhs)))
df.to_csv('brute_force_time.csv',index=False)