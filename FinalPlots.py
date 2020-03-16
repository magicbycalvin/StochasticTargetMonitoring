#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 02:21:24 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt

from CreatePlots import main

plt.close('all')
figure, axes = plt.subplots(2, 3)

main(axes[0, 0], 10)
axes[0, 0].set_title('t = 10')
main(axes[0, 1], 25)
axes[0, 1].set_title('t = 25')
main(axes[0, 2], 35)
axes[0, 2].set_title('t = 35')
main(axes[1, 0], 40)
axes[1, 0].set_title('t = 40')
main(axes[1, 1], 45)
axes[1, 1].set_title('t = 45')
main(axes[1, 2], 60)
axes[1, 2].set_title('t = 60')
