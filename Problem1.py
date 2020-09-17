# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 01:02:57 2020

@author: krish
"""
import math
from scipy.stats import norm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# The values from the question
# =============================================================================
mean = 0
sigma = 2 / math.sqrt(10)
sampleDraw = 100000
sns.set(style="ticks")
# =============================================================================
# Problem 1
# =============================================================================

x_max = round(norm.ppf(0.95, loc=mean, scale=sigma),2) 
#print (x_max)

# =============================================================================
# Problem 2
# =============================================================================

#def DrawSample(NumberOfDraws, mean, sigma, sampleSize):
#    Sample = []
#    for item in range(0, NumberOfDraws):
#        sample=np.random.normal(mean, sigma, sampleSize)
#        Sample.append(np.mean(sample))
#    return Sample 

#GeneratedSample = np.random.normal(mean, sigma, sampleDraw)
#PDFplot = sns.distplot(GeneratedSample,kde_kws={"shade":True}, hist=False, color = "c", label = "PDF of decision metric" )
#plt.axvline(x_max, color = "r", label = "Alarm bounds")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.savefig("PDF of decision metrics", dpi = 150)

# =============================================================================
# Problem 3  
# =============================================================================
HealthyState = []
FalseAlarm = []
NoOfTrials = 100000
for trials in range (0,NoOfTrials):
    nsamp= np.random.normal(mean, sigma, 100)
    for item in nsamp:
        if item > x_max:
            FalseAlarm.append(item)
        else:
            HealthyState.append(item)
print ("Number of Healthy Samples :",round(len(HealthyState)/NoOfTrials))
print("Number of False Alarm :",round(len(FalseAlarm)/NoOfTrials))



















