# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 00:26:22 2020

@author: krish
"""

import math
from scipy.stats import norm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# =============================================================================
# Values from question
# =============================================================================
mean1 = 0
sigma1 = 2 / math.sqrt(10)
mean2 = 2
sigma2 = 2 / math.sqrt(10) 
sampleDraw = 100000
x_max = round(norm.ppf(0.95, loc=mean1, scale=sigma1),2) 

# =============================================================================
# Problem 4
# =============================================================================

#SamplesFromHealthyState = np.random.normal(mean1, sigma1, sampleDraw)
#SamplesFromFaultyModel = np.random.normal(mean2, sigma2, sampleDraw)
#HealthyStatePlot = sns.distplot(SamplesFromHealthyState,kde_kws={"shade":True}, hist=False, color = "teal", label = "PDF of Healthy state" )
#FaultyStatePlot = sns.distplot(SamplesFromFaultyModel,kde_kws={"shade":True}, hist=False, color = "orange", label = "PDF of Faulty model" )
#plt.axvline(x_max, color = "r", label = "Alarm bounds")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.savefig("PDF of healthy and faulty model", dpi = 150)

#HealthyState = []
#FalseAlarm = []
#FaultyState = []
#NonDetectedFaults = []
#NoOfTrials = 100000
#for trials in range (0,NoOfTrials):
#    nsampHealthy= np.random.normal(mean1, sigma1, 100)
#    nsampFaulty= np.random.normal(mean2, sigma2, 100)    
#    for item in nsampHealthy:
#        if item > x_max:
#            FalseAlarm.append(item)
#        else:
#            HealthyState.append(item)
#    for item in nsampFaulty:
#        if item < x_max:
#            NonDetectedFaults.append(item)
#        else:
#            FaultyState.append(item)
#print ("Number of non- detected faults :",round(len(NonDetectedFaults)/NoOfTrials))
#print("Number of False Alarm :",round(len(FalseAlarm)/NoOfTrials))

# =============================================================================
# Problem 5
# =============================================================================
Lambda_np = math.exp((5*x_max)-5)
#print(Lambda_np)

# =============================================================================
# Problem 6
# =============================================================================

alphaValues = np.zeros(500)
betaValues = np.zeros(500)
decisionParameter = np.linspace(0.1,30,500)

for item in range (0,500):
    alphaValues[item] =1 - norm.cdf((0.2*math.log(decisionParameter[item])+1), mean1,sigma1) 
    betaValues[item] =norm.cdf((0.2*math.log(decisionParameter[item])+1), mean2,sigma2) 

#plt.plot(decisionParameter, alphaValues, color = "b", label = "False Alarm (α)" )
#plt.plot(decisionParameter, betaValues, color = "r", label = "Non-detection (β)" )
#plt.xlabel("Decision Parameter λ_np")
#plt.ylabel("Alpha/Beta Probabilities")
#plt.legend()
#plt.savefig("Alpha and Beta Vs Decision Parameter", dpi = 150)
#plt.show()

# =============================================================================
# Problem 7
# =============================================================================

ROC = 1 - betaValues 

plt.plot(alphaValues, ROC, color = "b", label = "ROC")
plt.legend()
plt.savefig("ROC", dpi = 150)
plt.show()


















