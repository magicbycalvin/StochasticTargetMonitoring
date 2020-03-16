#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:37:58 2020

@author: ckielasjensen
"""

import numpy as np
from scipy.integrate import solve_ivp


class Target:
    """ Target class

    state:
        [0] - X
        [1] - Y
        [2] - Psi
        [3] - V
        [4] - W
    """
    def __init__(self, x0, y0, psi0, v0=0.0, w0=0.0):
        self.state = np.array([x0, y0, psi0, v0, w0], dtype=float)

        self.t = 0.0

        self._last_time = 0.0

    def get_state(self):
        """ Gets the current state of the target

        :return: State vector in the form of [x, y, psi, v, w] where (x, y) is
            the 2D position of the vehicle, psi is the current heading, v is
            the current speed in the x-direction, and w is the angular rate.
        :rtype: np.ndarray
        """
        return np.copy(self.state)

    def send_cmd(self, v, w):
        """ Sends speed and angular rate commands to the target

        This assumes that the target perfectly matches the commanded speed and
        angular rates

        :param v: Commanded speed
        :type v: float
        :param w: Commanded angular rate
        :type w: float
        """
        self.state[3] = v
        self.state[4] = w

    def update(self, t):
        """ Updates the time and then state of the agent

        :param t: Time at which to update the agent
        :type t: float
        """
        self.update_time(t)
        self.update_state()

    def update_time(self, t):
        """ Updates the internal time of the agent

        :param t: Time at which to update the agent
        :type t: float
        """
        self._last_time = self.t
        self.t = t

    def update_state(self):
        """ Updates the state of the agent
        """
        dt = self.t - self._last_time
        self.state[0] += dt*self.state[3]*np.cos(self.state[2])
        self.state[1] += dt*self.state[3]*np.sin(self.state[2])
        self.state[2] += dt*self.state[4]

#        y0 = self.state[:3]
#        v = self.state[3]
#        w = self.state[4]
#        def fn(t, x): return [v*np.cos(x[2]), v*np.sin(x[2]), w]
#
#        sol = solve_ivp(fn, (self._last_time, self.t), y0)
#
#        self.state[:3] = sol.y[:, -1].copy()


#        dt = self.t - self._last_time
#        self.x += dt*self.v*np.cos(self.psi)
#        self.y += dt*self.v*np.sin(self.psi)
#        self.psi += dt*self.w
