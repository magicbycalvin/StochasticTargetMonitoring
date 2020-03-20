#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 06:42:06 2020

@author: ckielasjensen
"""

import numpy as np


class Parameters:
    """
    """
    def __init__(self):
        # Agent
        self.nveh = 3       # Number of vehicles
        self.dsafe = 1      # Minimum safe distance between vehicles (m)
        self.vmax = 100      # Maximum speed (m/s)
        self.vmin = 1       # Minimum speed (m/s)
        self.wmax = np.pi/2     # Maximum angular rate (rad/s)
        self.monSpeed = 3.0

        # Target constraints
        self.outerR = 125
        self.innerR = 50
        self.noflyR = 10
        self.detPer = 1     # Detection period of the target (s)

        # Optimization constraints
        self.deg = 5       # Order of approximation
        self.degElev = 10
        self.tflight = 30.0     # Flight traj time (s)
        self.tmon = 10.0    # Monitoring traj time (s)

        # Misc
        np.random.seed(0)
        self.iprint = 0     # Verbosity of minimizer output (0, 1, or 2)
        self.relaxation = 1e-3  # Relaxation for final mon point and inner R
        self.replanRad = 5  # If Ept is this different, replan
