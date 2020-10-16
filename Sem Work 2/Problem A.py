# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
import matplotlib.pyplot as plt

dataset = pd.read_csv('Dataset.csv', sep=",")
#sns.regplot(x="Time", y="Deviation", data=dataset, ci=None, line_kws={'color':'red'},scatter_kws={"s": 50, "alpha": 0.75})

t = dataset.Time
Y = dataset.Deviation

# =============================================================================
# Problem 2
# =============================================================================
# =============================================================================
#print ("Considering C=0")
# =============================================================================

numerator = 0
denominator = 0
for item in range(0,len(t)):
    numerator += t[item] *Y[item]
    denominator += t[item]**2
a = numerator / denominator
print ("The value of parameter a is:",round(a,4))
plt.scatter(t,Y, color = "teal",label='Observed values Y(t)')
plt.plot(t,a*t, color = "red",label='Fitted line X(t)')
#plt.legend(frameon=False, loc='lower center', ncol=2)
plt.xlabel("Time")
plt.ylabel("Values")
#plt.savefig("Problem 2",dpi =150)
plt.close()
# =============================================================================
#print ("Without considering C=0")
# =============================================================================

#stats = linregress(t, Y)
#m = stats.slope
#b = stats.intercept
#print("The value of parameter a is:",round(m,3))
#print("The value of parameter c is:",round(b,3))
#plt.scatter(t,Y, color = "darkturquoise",label='Observed values Y(t)')
#plt.plot(t,m * t + b, color = "red",label='Fitted line X(t)')
#plt.legend(frameon=False, loc='lower center', ncol=2)
#plt.xlabel("Time")
#plt.ylabel("Values")
#plt.savefig("Problem 2_Appendix", dpi = 150)
#plt.close()

# =============================================================================
# Problem 3
# =============================================================================

Y_fitted = a*t
error = Y - Y_fitted
MeanOfErrors = np.mean(error)
StdevOfErrors = np.std(error)
print("The mean of errors :",round(MeanOfErrors,3))
print("The standard deviation of errors :",round(StdevOfErrors,3))

# =============================================================================
# Problem 4
# =============================================================================

T2 = np.linspace(60,120,60)
X_T2 = a*T2
SimPoints = []


for i in range (0,10000):
    error_T2 = np.random.normal(MeanOfErrors, StdevOfErrors, 60)
    Y_T2 = X_T2 + error_T2
#    plt.plot(T2, Y_T2 )
    SimPoints.append(Y_T2)
#plt.axhline(y=10, color = 'black',label = "Threshold", linestyle = "dotted")
#plt.legend()    
#plt.savefig("Problem 4", dpi = 150)
#plt.close()


# =============================================================================
# Problem 5
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
#sns.set_style("ticks")
#sns.distplot(RUL,bins = 30, color = 'teal') 
#plt.xlabel("Remaining Useful Life (RUL)")
#plt.ylabel("Count")
#plt.savefig("Problem 5", dpi = 150)
#plt.close()

# =============================================================================
# Problem 10.(a)            
# =============================================================================

count = 0
for element in RUL:
    if element < 34:
        count += 1
Probabiity = count/len(RUL)
print(" The P(RUL(t_j))<=34 =",round(Probabiity,3))
















