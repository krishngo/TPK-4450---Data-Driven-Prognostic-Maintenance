# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:38:19 2020

@author: krish
"""
import math
from scipy.stats import norm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# Values from previous section and the question
# =============================================================================
mean = 0
sigma = 2 / math.sqrt(10)
NoOfSample = 10
x_max = round(norm.ppf(0.95, loc=mean, scale=sigma),2) 
MeanTemperatures = np.linspace(0,2,21)
MeanForEachShift = []
DecisionAtTime = np.zeros(21)

for item in range(0,20):
    ValuesOfSample = np.random.normal(MeanTemperatures[item],sigma,NoOfSample)
    MeanAtTemp = sum(ValuesOfSample)/len(ValuesOfSample)
    MeanForEachShift.append(MeanAtTemp)
    
for item in range(0,len(MeanForEachShift)):
    if MeanForEachShift[item] >= x_max:
        DecisionAtTime[item] = 1
    else:
        DecisionAtTime[item] = 0

DecisionPlot = sns.scatterplot(x= np.linspace(0,20,21),y = DecisionAtTime, hue = DecisionAtTime, palette={0:"g",1:"r"} )
plt.xlabel("Shifts")
plt.ylabel("Accept (0) or Reject (1)")
plt.savefig("Problem 8", dpi=150)
