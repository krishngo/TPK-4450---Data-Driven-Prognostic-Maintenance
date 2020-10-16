# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:03:37 2020

@author: krish
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
import matplotlib.pyplot as plt
import math

# =============================================================================
# Problem6
# =============================================================================
dataset = pd.read_csv('Dataset.csv', sep=",")

T = dataset.Time
Y = dataset.Deviation
Increment = np.zeros(len(Y)-1)
for item in range(0,len(Y)-1):
    Increment[item] = Y[item+1] - Y[item] 
MeanOfIncrement = np.mean(Increment)
SDofIncrement = np.std(Increment)
print("Mean of the increments: ", round(MeanOfIncrement,3))
print("Standard Deviation of the increments: ", round(SDofIncrement,3))

# =============================================================================
# Problem 7
# =============================================================================

T2 = np.linspace(60,500,440)
First_point = 6.194306291
SimPoints = []
plt.plot(T,Y, color = "midnightblue",linewidth=3.0, label = "Observed Points")
for i in range (0,10000):
    RndIncrements = np.random.normal(MeanOfIncrement, SDofIncrement, 440)
    SumOfIncrements = np.cumsum(RndIncrements)
    Y_T2 = First_point + SumOfIncrements
    plt.plot(T2,Y_T2,color = "lightsteelblue")
    SimPoints.append(Y_T2)
plt.axhline(y=10, color = 'darkred',label = "Threshold",linestyle='dotted')
plt.legend(frameon=False, loc='upper center', ncol=2)
plt.savefig("Problem 7",dpi = 150)

# =============================================================================
# Problem 8
# =============================================================================

TimeofThresholdPassing = []

RUL = []
for items in range(0,len(SimPoints)):
    for item in range(0,len(SimPoints[items])):
        if SimPoints[items][item] > 10:
            TimeofThresholdPassing.append(T2[item])
            break
for element in TimeofThresholdPassing:
    element -= 60
    RUL.append(element)
#plt.hist(RUL,bins = 50, color = 'teal', density = True,label = "Calculated RUL distribution") 
#plt.xlabel("Remaining Useful Life (RUL)")
#plt.ylabel("Count")
#plt.savefig("Problem 8", dpi = 150)
#plt.close()

# =============================================================================
# Problem 9
# =============================================================================

mu_hat = ((10-First_point)/MeanOfIncrement)
lamda_hat = ((10-First_point)/SDofIncrement)**2
PDF_values = []

T_3 = np.linspace(0.0001,500,100000)
for item in range(0,100000):
    exponent = math.exp(-lamda_hat/(2*(mu_hat**2))*(((T_3[item]-mu_hat)**2)/T_3[item]))    
    f_t = math.sqrt(lamda_hat/(2*math.pi*(T_3[item]**3)))*exponent
    PDF_values.append(f_t)
#plt.plot(T_3,PDF_values,color = "red", label = "Theoretical PDF of RUL distribution")
#plt.legend()
#plt.savefig("Problem 9", dpi = 150)

# =============================================================================
# Problem 10.(a)            
# =============================================================================

count = 0
for element in RUL:
    if element < 34:
        count += 1
Probabiity = count/len(RUL)
print(" The P(RUL(t_j))<=34 =",round(Probabiity,3))

# =============================================================================
# Problem 10.b
# =============================================================================

count = 0
for element in RUL:
    if element < 34:
        count += 1
Probabiity = count/len(RUL)
print(" The P(RUL(t_j))<=34 =",round(Probabiity,3))

# =============================================================================
# Problem 10.c
# =============================================================================

def integrate_composite_trapezoidal_rule(func, lower_limit, upper_limit, n):
    dx = (upper_limit - lower_limit) / n
    I = 0
    for k in range(0,n):
        step_lower_limit = lower_limit + (dx * k)
        step_upper_limit = lower_limit + (dx * (k + 1))
        I += (step_upper_limit - step_lower_limit) * ((func(step_lower_limit) + func(step_upper_limit)) / 2)
    return I
def function(x):
    return math.sqrt(lamda_hat/(2*math.pi*(x**3)))* math.exp(-lamda_hat/(2*(mu_hat**2))*(((x-mu_hat)**2)/x))

Probability2 = integrate_composite_trapezoidal_rule(function,0.0001,34,20) 
print(round(Probability2,3))













    