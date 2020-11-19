# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:44:40 2020

@author: krish
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gamma
import math

Lambda = 17.8e-6 
MTTF = 5.6e4

# =============================================================================
# Time based maintenacne
# =============================================================================

C = 50
k= 500

# =============================================================================
# Problem 1
# =============================================================================

#t_c = np.linspace(730,87600,120)
#Cost = []
#
#def MeanCost(t_c):
#    MC = (C + (C+k)*(1-math.exp(-Lambda*t_c)))/t_c
#    return MC
#
#for element in t_c:
#    Cost.append(MeanCost(element))
#
#index = Cost.index(min(Cost))
#print (t_c[index])

# =============================================================================
# Problem 2 - clock
# =============================================================================
#t_c = np.linspace(10000,100000,5001)
#Cost = []
#def MeanCost(t_c):
#    F_t = 1-math.exp(math.pow(-1.59e-5*t_c,3))
#    MC = (C + (C+k)*F_t)/t_c
#    return MC
#
#for element in t_c:
#    Cost.append(MeanCost(element))
#index = Cost.index(min(Cost))
#print (t_c[index])
#plt.plot(t_c, Cost)
#plt.plot(t_c[index],Cost[index], marker='o', markersize=3, color="red", label = "Optimal Time")
#plt.ylabel("Asymptotic cost per interval")
#plt.xlabel("Time Interval (t_c)")
#plt.legend(frameon=False, loc='lower center', ncol=2)
#plt.savefig("Problem 2-clock", dpi = 150)
#plt.close

# =============================================================================
# Problem 3
# =============================================================================

dataset = pd.read_csv('cmonitoring.txt', sep=",", header = None)
dataset.columns = ["Unit 1", "Unit 2", "Unit 3", "Unit 4", "Unit 5", 
                   "Unit 6", "Unit 7", "Unit 8", "Unit 9", "Unit 10"]

#sns.lineplot(data=dataset, dashes=False)
#plt.xlabel("Test interval(τ)")
#plt.ylabel("Degradation")
#plt.legend()
#plt.savefig("Problem 3 point plot", dpi = 150)
#plt.close()
#
increment = dataset.diff().drop([0])
#for column in increment:
#    columnVal = increment[column]
#    sns.distplot(columnVal, label = column )
#plt.xlabel("Time")
#plt.ylabel("Frequency")
#plt.legend()
#plt.savefig("Problem 3 increments", dpi = 150)

# =============================================================================
# Problem 4
# =============================================================================

shape, loc, scale = gamma.fit(increment, floc=0)

#print("The shape parameter is given by:", round(shape,3))
#print("The location parameter is given by:", round(loc,3))
#print("The scale parameter is given by:", round(scale,3))


# =============================================================================
# Problem 6
# =============================================================================
trajectory = {}

for item in range(1, 1001):
    x = 0
    DegradationLevel = []    
    for element in range(40):
        x += gamma.rvs(shape, loc, scale)
        DegradationLevel.append(x)
    trajectory[f'T{item}'] = DegradationLevel
trajectories = pd.DataFrame(trajectory)

    
time = [i for i in range(0, 40*12, 12)]
#for i in range(1, 1001):
#    plt.plot(time, trajectories[f'T{i}'], color = "lightsteelblue")
#plt.axhline(y=75, color = 'darkred',label = "Maintenance Threshold",linestyle='dotted')
#plt.axhline(y=100, color = 'darkgreen',label = "Failure Threshold",linestyle='dashed')
#plt.xlabel("Weeks")
#plt.ylabel("Degradation Level")
#plt.legend(frameon=False, loc='upper center', ncol=2)
#plt.savefig("Problem 6",dpi = 150)
    
# =============================================================================
# Problem 7a
# =============================================================================

Lvalues =  []
time = [i for i in range(0, 40*12, 12)]
for i in range(1, 1001):
    if any(trajectories[f'T{i}'].values >= 100):
        Lvalues.append(np.interp(100, trajectories[f'T{i}'], time))
sigmaL = np.array(Lvalues)
#sns.distplot(sigmaL, bins=50, color = "Teal")
#plt.xlabel('Time')
#plt.ylabel('Frequency')
#plt.savefig("Problem 7a",dpi = 150)
#plt.show()

# =============================================================================
# Problem 7b
# =============================================================================

sigmaM =  []
time = [i for i in range(0, 40*12, 12)]
for i in range(1,1001):
    for j in range(len(trajectories['T1'])):
        if trajectories[f'T{i}'][j] >= 75:
            sigmaM.append(time[j])
            break
#sns.distplot(sigmaM, kde = False, bins=50, color = "Teal")
#plt.xlabel('Time')
#plt.ylabel('Frequency')
#plt.savefig("Problem 7b",dpi = 150)
#plt.show()

# =============================================================================
# Problem 8
# =============================================================================

MTTF = sigmaL.mean()

print ("MTTF = ",MTTF)

# =============================================================================
# Problem 9
# =============================================================================

t = 0
for i in range(len(sigmaL)):
    t += min(sigmaL[i], sigmaM[i])
MTBR = t/len(sigmaL)

print("MTBR =",MTBR)

# =============================================================================
# Problem 10
# =============================================================================

ci = 10
C = 50
K = 500
asymptoticCost = []
time = [i for i in range(0, 40*12, 12)]
for i in range(1,1001):
    if any(trajectories[f'T{i}'].values >= 100):
        sigmaL = np.interp(100, trajectories[f'T{i}'], time)
        k_L = sigmaL//12
    for j in range(len(trajectories[f'T{i}'])):
        if trajectories[f'T{i}'][j] >= 75:
            sigmaM = time[j]
            k_M = time[j]/12
            break
    if sigmaL > sigmaM:
        cost = (ci*k_M + C)/sigmaM
    else:
        cost = (ci*k_L + C + K)/sigmaL
    asymptoticCost.append(cost)
    
cost = np.array(asymptoticCost).mean()
print ("Asymptotic Cost per unit time =", cost)

# =============================================================================
# Problem 11 & 12
# =============================================================================

def AsymptoticCost(trajectories,M):
    ci = 10
    C = 50
    K = 500
    asymptoticCost = []
    time = [i for i in range(0, 40*12, 12)]
    for i in range(1,1001):
        if any(trajectories[f'T{i}'].values >= 100):
            sigmaL = np.interp(100, trajectories[f'T{i}'], time)
            k_L = sigmaL//12
        for j in range(len(trajectories[f'T{i}'])):
            if trajectories[f'T{i}'][j] >= M:
                sigmaM = time[j]
                k_M = time[j]/12
                break
        if sigmaL > sigmaM:
            cost = (ci*k_M + C)/sigmaM
        else:
            cost = (ci*k_L + C + K)/sigmaL
        asymptoticCost.append(cost)
    return np.array(asymptoticCost).mean()

Cost = []
M = [i for i in range(75,96)]
for i in range(75, 96):
    Cost.append(AsymptoticCost(trajectories, i))

sns.lineplot(M, Cost)
OptimumThreshold = M[Cost.index(min(Cost))]
print("Optimum Threshold:",OptimumThreshold)
plt.axvline(OptimumThreshold, color='red', linestyle='dotted', label = "Optimum Threshold")
plt.xlabel('M')
plt.ylabel('Asymptotic cost per unit of time')
plt.legend()
plt.savefig("Problem 11",dpi = 150)
plt.show()    


















