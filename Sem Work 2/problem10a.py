"""
title:          problem5
version:        1.0.0
fileName:       problem5.py 
author:         Joachim Nilsen Grimstad
description:    problem 5, semester work II, TPK–4450, Autumn 2020 @NTNU                                
license:        GNU General Public License v3.0 https://www.gnu.org/licenses/gpl-3.0.en.html 
                
disclaimer:     Author takes no responsibility for any use other than authors own evaluation in courses at NTNU.
""" 

# Dependancies 
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from math import sqrt

#from functions import ImportFile
from scipy.interpolate import interp1d

# Seed
np.random.seed(4450)

# Data
dataset = pd.read_csv('Dataset.csv', sep=",")
t = dataset.Time
y = dataset.Deviation

# Least square estimation
t_squared = []                                                                          # Empty set of x^2
yt = []                                                                                 # Empty set of xy
N = len(t)
y_ans = []
error = []                                                                              # Number of observations
for i in range(len(t)):                                                                 # For all observations, do:
    t_squared.append(t[i] ** 2)                                                             # x^2, save in list x_squared
    yt.append(t[i] * y[i])                                                                  # xy, save in list xy
a = sum(yt) / sum(t_squared)  
                                                         # Slope
for i in range(len(t)):                                                                 # For all observations, do:
    y_ans.append(a*t[i])                                                                    # a * x, save to y_ans
    error.append(y[i] - y_ans[i])                                                           # y(t) - X(t), save to error

# Maximum Likelihood Estimation                                                         
μ_hat = 1/N * sum(error)                                                                # μ_hat                                                                                                                                   
s = []                                                                                  # empty set of (errors - u_hat)^2
for i in range (len(t)):                                                                # For all observations, do:
    s.append((error[i] - μ_hat)**2)                                                      # (errors - u_hat)^2, save to error
var_hat = 1/N * sum(s)                                                                  # var_hat
sd_hat = sqrt(var_hat)                                                                  # sd_hat

#Trajectories
y_trajectories = []                                                                     # List of y values for trajectories
t_trajectory = list(range(60, 120 + 1))                                                 # List of the x values for the trajectories        
counter = 0                                                                             # Counter to stop while loop
while counter < 1e4:                                                                    # while loop until counter is 10 000
    y_trajectory = []                                                                   # y values for one trajectory
    for i in range(60, 120 + 1):                                                        # for loop from 60 to 120.
        if i == 60: 
            y_trajectory.append(y[i])                                                  # adds first point
        else:   
            error = np.random.normal(μ_hat, sd_hat, 1)                                 # Normally distributed error term
            y_trajectory.append(a * i + error[0])                                       # Y - values
    counter += 1                                                                        # Increase counter
    y_trajectories.append(y_trajectory)                                                 # save the generated trajectory

threshold_t = []
for i in range(len(y_trajectories)):
    f = interp1d(y_trajectories[i], t_trajectory)
    y_theshold_t = f(10)    
    threshold_t.append(y_theshold_t)
    print(threshold_t)
#for items in range(len(y_trajectories)):
#    for item in range(0,len(y_trajectories[items])):
#        if y_trajectories[items][item] > 10:
#            threshold_t.append(t_trajectory[item])
#            break
RUL = []
for i in range(len(threshold_t)):
    RUL.append(threshold_t[i] - 60)

count = 0
for element in RUL:
    if element < 34:
        count += 1
prob = count / len(RUL)
print(prob)