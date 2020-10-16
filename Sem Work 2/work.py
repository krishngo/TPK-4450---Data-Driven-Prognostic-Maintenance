# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 00:19:54 2020

@author: krish
"""

import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

x = np.arange(0, 10)
y = np.exp(x)
f = interpolate.interp1d(x, y)

#xnew = np.arange(0, 9, 0.1)
ynew = f(5)   # use interpolation function returned by `interp1d`
print(ynew)
#plt.plot(x, y, 'o', xnew, ynew, '-')
#plt.show()