#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:47:01 2020

@author: ckielasjensen
"""

import numpy as np
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation as R


class DubinsDynamics:
    """Dynamics for a Dubin's car model

    x0 (float) - Initial global x position. Optional, default 0.0
    y0 (float) - Initial global y position. Optional, default 0.0
    v0 (float) - Initial speed in the x direction of the body frame. Optional,
        default 0.0
    psi0 (float) - Initial heading of the body. Optional, default 0.0
    w0 (float) - Initial angular rate. Optional, default 0.0
    state_freq (float) - Frequency at which the update the internal state (Hz),
        default 100.0
    """
    def __init__(self, x0=0.0, y0=0.0, v0=0.0, psi0=0.0, w0=0.0,
                 state_freq=100.0):
        # Initialize variables
        self.x = x0
        self.y = y0
        self.v = v0
        self.psi = psi0
        self.w = w0

        self._cmdX = 0
        self._cmdY = 0
        self._cmdPsi = 0
        self._cmdV = 0
        self._cmdW = 0

    def updateState(self, *args, **kwargs):
        """Updates the current state of the Dubin's car model

        For now, this function assumes that the target perfectly follows the
        commanded position.

        Return (float) - Time at which the state was updated.
        """
        now = time.Time()
        result = odeint(
                        self.model,
                        [self.x, self.y, self.psi],
                        [self._lastTime, now])
        self._lastTime = now

        self.x = result[-1, 0]
        self.y = result[-1, 1]
        self.psi = result[-1, 2]
        self.v = self._cmdV
        self.w = self._cmdW

        return now


    def model(self, x, t):
        """
        """
        xpos = x[0]
        ypos = x[1]
        psi = x[2]

        f = [
            self._cmdV*np.cos(psi),
            self._cmdV*np.sin(psi),
            self._cmdW
            ]

        return f
