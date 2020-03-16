#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:38:11 2020

@author: ckielasjensen
"""

import numpy as np

from agent import Agent
from target import Target


class Parameters:
    """
    """
    def __init__(self):
        self.deg = 7        # Order of approximation
        self.nveh = 3       # Number of vehicles
        self.ndim = 2       # Number of dimensions
        self.dsafe = 1      # Minimum safe distance between vehicles (m)
        self.odsafe = 2     # Minimum safe distance from obstacles (m)
        self.vmax = 10      # Maximum speed (m/s)
        self.wmax = np.pi/2  # Maximum angular rate (rad/s)
        self.tf = 25

        # Target constraints
        self.outerR = 10
        self.innerR = 5
        self.noflyR = 2

        self.iniPt = np.array([0, 0])
        self.iniSpeed = 3
        self.iniAng = 0
        self.monSpeed = 3
        self.monT = (self.outerR-self.innerR)/self.monSpeed


class Simulator:
    """
    """
    def __init__(self):
        pass

    def time(self):
        return self._time
