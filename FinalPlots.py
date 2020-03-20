#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 02:21:24 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt

from CreatePlots import main

plt.rcParams.update({
            'font.size': 20,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'lines.linewidth': 2,
            'lines.markersize': 9
            })

#plt.rcParams.update(plt.rcParamsDefault)
#plt.ion()

plt.close('all')
figure, axes = plt.subplots(3, 2)

main(axes[0, 0], 0)
axes[0, 0].set_title('t = 0')
main(axes[0, 1], 10)
axes[0, 1].set_title('t = 10')
main(axes[1, 0], 30)
axes[1, 0].set_title('t = 30')
main(axes[1, 1], 45)
axes[1, 1].set_title('t = 45')
main(axes[2, 0], 50)
axes[2, 0].set_title('t = 50')
main(axes[2, 1], 60)
axes[2, 1].set_title('t = 60')

# Remove all but the first legend
legs = []
for i in axes:
    for j in i:
        legs.append(j.get_children()[-2])
for leg in legs[1:]:
    leg.set_visible(False)

## Make the past trajs red
#lines = axes[1, 1].get_children()
#for i, line in enumerate(lines):
#    try:
#        print(f'i: {i}, color: {line.get_color()}')
#    except Exception:
#        pass
#lines[8].set_color('r')
#lines[2].set_color('r')
#
#print('---')
#
#lines = axes[1, 2].get_children()
#for i, line in enumerate(lines):
#    try:
#        print(f'i: {i}, color: {line.get_color()}')
#    except Exception:
#        pass
#lines[8].set_visible(False)
#lines[2].set_visible(False)
