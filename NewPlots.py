#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:10:37 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt

from Mission1 import main as main1
from Mission2 import main as main2
from Mission3 import main as main3


if __name__ == '__main__':
    plt.close('all')

    # Mission 1 plots
    _, m1ret = main1()
    agents1 = m1ret[2]
    for ag in agents1:
#        ag._flight_plot[0].set_visible(False)
        ag._flight_plot[0].set_alpha(0.2)
#        ag._flight_plot[0].set_visible(True)
#        ag._flight_plot[1].set_alpha(1)

    # First Mission 2 plots
    _, m2ret = main2(2.0)
    agents2 = m2ret[2]
    for ag in agents2:
        ag._flight_plot[0].set_alpha(0.2)

    # Second Mission 2 plots
    _, m2ret = main2(30.0 + 2*10, targetTracer=True)
    agents2 = m2ret[2]
    agents2[0]._flight_plot[0].set_visible(False)
    agents2[0]._flight_plot[1].set_alpha(0.2)
    agents2[1]._flight_plot[0].set_visible(False)
    agents2[1]._flight_plot[1].set_alpha(0.2)
    agents2[2]._flight_plot[0].set_alpha(0.2)

    # Third Mission 2 plots
    _, m2ret = main2()
    agents2 = m2ret[2]
    for ag in agents2:
        ag._flight_plot[0].set_visible(False)
        ag._flight_plot[1].set_alpha(0.2)

    # Mission 3 plots
    _, m3ret = main3()
    agents3 = m3ret[2]
    for ag in agents3:
        ag._flight_plot[0].set_visible(False)
        ag._flight_plot[1].set_alpha(0.2)
