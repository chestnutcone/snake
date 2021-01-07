# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:59:29 2020

@author: Oliver
"""

import matplotlib.pyplot as plt
import numpy as np
import time

# x = np.linspace(0, 6*np.pi, 100)
# y = np.sin(x)

# # You probably won't need this if you're embedding things in a tkinter plot...
# plt.ion()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

# for phase in np.linspace(0, 10*np.pi, 500):
#     line1.set_ydata(np.sin(x + phase))
#     fig.canvas.draw()
#     fig.canvas.flush_events()


fig1, ax1 = plt.subplots()

arr = np.zeros((10,10))
axim1 = ax1.imshow(arr)

for i in range(10):
    arr[i,i] = 1
    print(arr)
    axim1.set_data(arr)
    fig1.canvas.flush_events()
    time.sleep(1)