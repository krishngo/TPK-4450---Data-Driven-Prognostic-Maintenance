# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:12:03 2020

@author: krish
"""
# =============================================================================
# example of exponential distribution for the population is used
# =============================================================================
# =============================================================================
# Imported modules
# =============================================================================
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
# =============================================================================
# Code
# =============================================================================
def GenerateNormalDistributionPopulation(mean, sigma, Size):
    s = np.random.normal(loc=mean, scale=sigma, size= Size)
    return s

def GeneratePlot(SampleMean, NumberOfSampling):
    plt.subplots(1, 1, figsize=(10,5))
    plt.hist(SampleMean, 200, density=True, label ="Sample distribution")
    plt.title("Number of samples :" + str(NumberOfSampling))
    x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
    # draw standard normal disrtibution line
    plt.plot(x, stats.norm.pdf(x, mean, sigma),linewidth = 2, color='red', label ="Normal population distribution")
    plt.legend()
    plt.savefig('CLT for N-'+str(NumberOfSampling))   
        
def GenerateCentralLimitTheoremProof(mean, sigma, sampleSize):
    population = GenerateNormalDistributionPopulation(mean, sigma, 100000)
    SampleMean = []
    NumberOfSampling = [1000, 10000, 50000] #random sample sizes
    SampleSize = sampleSize
    for item in NumberOfSampling:
        MeanOfSample = []
        for value in range(0,item):
            RandomSelectedSamples = random.choices(population, k = SampleSize)
            Mean = sum(RandomSelectedSamples)/len(RandomSelectedSamples)
            MeanOfSample.append(Mean)
        SampleMean.append(MeanOfSample)
    GeneratePlot(SampleMean[0], NumberOfSampling[0])
    GeneratePlot(SampleMean[1], NumberOfSampling[1])
    GeneratePlot(SampleMean[2], NumberOfSampling[2])
    CalculateSampleMeanAndStandardDeviation(SampleMean[0], NumberOfSampling[0], mean, sigma, SampleSize)
    CalculateSampleMeanAndStandardDeviation(SampleMean[1], NumberOfSampling[1], mean, sigma, SampleSize)
    CalculateSampleMeanAndStandardDeviation(SampleMean[2], NumberOfSampling[2], mean, sigma, SampleSize)
    
    

def CalculateSampleMeanAndStandardDeviation(SampleMean, NumberOfSampling, mean, sigma, SampleSize):
    print("Statistics for Number of draws="+ str(NumberOfSampling))
    print("expected value of sample:", round(np.mean(SampleMean),2))
    print("standard deviation of sample:", round(np.std(SampleMean),2))
    print("standard deviation of population divided by squre root of sample size:", round(sigma/np.sqrt(SampleSize),2))
    print("Deviation =", round(np.std(SampleMean)-(sigma/np.sqrt(SampleSize)),3))
          
# =============================================================================
# Main    
# =============================================================================
mean = 0
sigma = 2 / math.sqrt(10)
GenerateCentralLimitTheoremProof(0,2, 10)
   
    