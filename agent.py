#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:37:50 2020

@author: ckielasjensen
"""
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize

from bezier import Bezier
from planner import plan_flight, plan_mon


# TODO
#   1. Create Vehicle parent class and have agent and target inherit from it
#   2. Use an ODE solver to find the expected position of the target


class Agent:
    """ Agent class

    state:
        [0] - X
        [1] - Y
        [2] - Psi
        [3] - V
        [4] - W
    """
    agentIdx = 0
    trajList = []
    timeList = []
#    flightTrajDict = OrderedDict()
#    flightTimesDict = OrderedDict()
#    monTrajDict = OrderedDict()
#    monTimesDict = OrderedDict()
    colors = ['c', 'g', 'b']

    def __init__(self, x0, y0, psi0, v0, w0, params, ax=None):
        self.state = np.array([x0, y0, psi0, v0, w0], dtype=float)
        self.params = params

        self.flight_traj = None
        self.mon_traj = None
        self.trgt_state = None
        self.t = 0.0
        self.idx = Agent.agentIdx
        Agent.agentIdx += 1

        self._last_time = 0.0
#        self._traj_list = np.atleast_2d([])
#        self._time_list = np.atleast_2d([])
        self._traj_state = 'flight'
        self._ax = ax
        self._last_trgt = None
        self._flight_plot = None
        self._mon_plot = None
        self._arrow = None

    # TODO
    #   * Fix the dt=None since it is a messy workaround for adding differently
    #     timed trajectories
    def compute_flight_traj(self, tf=None):
        """ Plans the flight trajectory
        """
        print(f'Agent {self.idx} computing flight')
        self._traj_state = 'flight'

        # Grab states
        p0 = self.state[:2]
        v0 = self.state[3]
        psi0 = self.state[2]
        t0 = self.t
        try:
            pastCpts = np.vstack(Agent.trajList)
            pastTimes = np.vstack(Agent.timeList)
        except ValueError:
            pastCpts = np.atleast_2d([])
            pastTimes = np.atleast_2d([])

        if tf is None:
            tf = t0 + self.params.tflight

        # Predict target position and trajectory and save the prediction
        trgt = self.predict_target(tf)
        trgt_cpts = self.predict_trgt_traj(tf)
        self._last_trgt = trgt.copy()

        trgt_traj = Bezier(trgt_cpts, t0=t0, tf=tf)
#        trgt_traj.plot(self._ax, showCpts=False, color='k', ls=':')

        # Plan the flight trajectory and then share it to the other agents via
        # the Agent class variable
        flight_traj = plan_flight(p0, v0, psi0, t0, trgt, trgt_cpts, pastCpts,
                                  pastTimes, self.params, tf=tf)
        self.flight_traj = flight_traj
        Agent.trajList.append(flight_traj.cpts)
        Agent.timeList.append([flight_traj.t0, flight_traj.tf])
        self.flightTrajIdx = len(Agent.trajList) - 1
#        Agent.flightTrajDict[self.idx] = flight_traj.cpts.squeeze()
#        Agent.flightTimesDict[self.idx] = [flight_traj.t0, flight_traj.tf]

        # If we have an axis, plot the new trajectory
        if self._ax is not None:
            if self._flight_plot is None:
                self._flight_plot = self._ax.plot(flight_traj.curve[0, :],
                                                  flight_traj.curve[1, :],
                                                  color=Agent.colors[self.idx],
                                                  linestyle='-')[0]
            else:
                self._flight_plot.set_xdata(flight_traj.curve[0, :])
                self._flight_plot.set_ydata(flight_traj.curve[1, :])
#            flight_traj.plot(self._ax, showCpts=False,
#                             color=Agent.colors[self.idx], linestyle='-')
            plt.pause(0.001)

    def compute_mon_traj(self, tf=None):
        """ Plans the monitoring trajectory
        """
        print(f'Agent {self.idx} computing mon')
        self._traj_state = 'mon'

        # Grab states
        pdot = self.flight_traj.diff()
        p0 = self.flight_traj.cpts[:, -1]
        v0 = np.linalg.norm(pdot.cpts[:, -1])
        psi0 = np.arctan2(pdot.cpts[1, -1], pdot.cpts[0, -1])
        t0 = self.flight_traj.tf
        try:
            pastCpts = np.vstack(Agent.trajList)
            pastTimes = np.vstack(Agent.timeList)
        except ValueError:
            pastCpts = np.atleast_2d([])
            pastTimes = np.atleast_2d([])

        if tf is None:
            tf = t0 + self.params.tmon

        # Predict the target's trajectory
        trgt_cpts = self.predict_trgt_traj(tf)
#        temp = Bezier(trgt_cpts, t0=t0, tf=tf)
#        temp.plot(self._ax, showCpts=False, linestyle=':')

        # Plan the monitoring trajectory and then share it to the other agents
        # via the Agent class variable
        mon_traj = plan_mon(p0, v0, psi0, t0, trgt_cpts, pastCpts, pastTimes,
                            tf, self.params)
        self.mon_traj = mon_traj
        Agent.trajList.append(mon_traj.cpts)
        Agent.timeList.append([mon_traj.t0, mon_traj.tf])
#        Agent.monTrajDict[self.idx] = mon_traj.cpts.squeeze()
#        Agent.monTimesDict[self.idx] = [mon_traj.t0, mon_traj.tf]

        # If we have an axis, plot the new trajectory
        if self._ax is not None:
            if self._mon_plot is None:
                self._mon_plot = self._ax.plot(mon_traj.curve[0, :],
                                               mon_traj.curve[1, :],
                                               color=Agent.colors[self.idx],
                                               linestyle='--')[0]
            else:
                self._flight_plot.set_xdata(mon_traj.curve[0, :])
                self._flight_plot.set_ydata(mon_traj.curve[1, :])
#            mon_traj.plot(self._ax, showCpts=False,
#                             color=Agent.colors[self.idx], linestyle='--')
            plt.pause(0.001)

    def detect_target(self, target_state):
        """ Detects the targets current state

        :param target_state: State vector of the target in the form
            [x, y, psi, v, w]
        :type target_state: np.ndarray
        """
        self.trgt_state = target_state
        if self._last_trgt is None:
            return

        # If we are in the flight state and the target goes out of our
        # prediction area, replan the trajectory
        if self._traj_state == 'flight':
            Ept = self.predict_target(self.flight_traj.tf)
            if np.linalg.norm(self._last_trgt - Ept) > self.params.replanRad:
                print(f'======= Agent {self.idx} =======')
                print(f'======= EPT REPLANNING! =======')
                print(f'tf: {self.flight_traj.tf}, t: {self.t}')
                print(f'Last Pos: {self._last_trgt}, Cur: {Ept}')
                Agent.trajList.pop(self.flightTrajIdx)
                self.compute_flight_traj(tf=self.flight_traj.tf)

    def predict_target(self, tf):
        """Predicts the future location of the target

        Ept - Expectation of the target position, pt
        """
        y0 = self.trgt_state[:3]
        v = self.trgt_state[3]
        w = self.trgt_state[4]

        def fn(t, x): return [v*np.cos(x[2]), v*np.sin(x[2]), w]

        sol = solve_ivp(fn, (self.t, tf), y0)
#        print(f'Target Prediction, t0: {self.t}, tf: {tf}')
#        print(f'State: {self.trgt_state}, Pred: {sol.y[:2, -1]}')

        Ept = np.array([sol.y[0, -1], sol.y[1, -1]])

#        print('agent.py - predict_target')
#        print(f'Ept: {Ept}')
        return Ept

    def predict_trgt_traj(self, tf):
        """Predicts the target's trajectory and fits it with a Bernstein poly
        """
        npts = self.params.deg + self.params.degElev + 1
        y0 = self.trgt_state[:3]
        v = self.trgt_state[3]
        w = self.trgt_state[4]

        def fn(t, x): return [v*np.cos(x[2]), v*np.sin(x[2]), w]

        sol = solve_ivp(fn, (self.t, tf), y0,
                        t_eval=np.linspace(self.t, tf, npts))

        trgt_cpts = sol.y[:2, :]

#        print('agent.py - predict_trgt_traj')
#        print(f'trgt_cpts: {trgt_cpts}')
        return trgt_cpts

    def get_state(self):
        """ Gets the current state of the target

        :return: State vector in the form of [x, y, psi, v, w] where (x, y) is
            the 2D position of the vehicle, psi is the current heading, v is
            the current speed in the x-direction, and w is the angular rate.
        :rtype: np.ndarray
        """
        return np.copy(self.state)

    def update(self, t):
        """ Updates the time and then state of the agent

        :param t: Time at which to update the agent
        :type t: float
        """
        self.update_time(t)
        self.replan_check()
        self.update_state()

    def update_time(self, t):
        """ Updates the internal time of the agent

        :param t: Time at which to update the agent
        :type t: float
        """
        self._last_time = self.t
        self.t = t

    def replan_check(self):
        """Determines whether it is time to plan a new trajectory
        """
        # Check whether its time to plan a flight trajectory
        if self._traj_state == 'mon':
            if self.t >= self.mon_traj.tf - 1e-9:
                self.compute_flight_traj()

        # Check whether its time to plan a monitoring trajectory
        elif self._traj_state == 'flight':
            if self.t >= self.flight_traj.tf - 1e-9:
                self.compute_mon_traj()

        # Note, we check for Ept replanning within the detect target method

    def update_state(self):
        """ Updates the state of the agent

        Uses simple linear quadrature integration to solve the Dubin's car
        dynamics, i.e.,
            x += dt*v*cos(psi)
            y += dt*v*sin(psi)
            psi += dt*w
        """
        # Actual dynamics
#        dt = self.t - self._last_time
#        self.state[0] += dt*self.state[3]*np.cos(self.state[2])
#        self.state[1] += dt*self.state[3]*np.sin(self.state[2])
#        self.state[2] += dt*self.state[4]

        # For now, we assume that the agent perfectly follows the trajectory
        traj = self.get_traj_cmd()
        trajdot = traj.diff()
        trajddot = trajdot.diff()

        t = self.t
        xdot = trajdot.x
        ydot = trajdot.y
        xddot = trajddot.x
        yddot = trajddot.y
        trajDotNormSqr = trajdot.normSquare()(t)

        self.state[0] = traj.x(t)
        self.state[1] = traj.y(t)
        self.state[2] = np.arctan2(ydot(t), xdot(t))
        self.state[3] = np.sqrt(trajDotNormSqr)
        if trajDotNormSqr == 0:
            self.state[4] = 0.0
        else:
            self.state[4] = ((xdot(t)*yddot(t) - xddot(t)*ydot(t)) /
                             trajDotNormSqr)

    def plot_arrow(self):
        """
        """
        if self._ax is not None:
            if self._arrow is not None:
                self._arrow.remove()

            x = self.state[0]
            y = self.state[1]
            dx = 3*np.cos(self.state[2])
            dy = 3*np.sin(self.state[2])
            self._arrow = self._ax.arrow(x, y, dx, dy,
                                         fc=Agent.colors[self.idx],
                                         ec='k',
                                         lw=1.5,
                                         width=0.75,
                                         zorder=100)

    def get_traj_cmd(self):
        """ Gets the current trajectory command

        Since there are two trajectories (flight and monitoring), we need to
        check which one to use.

        :return: Current trajectory
        :rtype: Bezier
        """
#        if self.flight_traj is None or self.mon_traj is None:
#            err = ('The flight and/or monitoring trajectory has not been '
#                   'initialized.')
#            raise Exception(err)
        # TODO put this in a try catch so that we still raise if we try to
        # get something uninitialized
        if self.t >= self.flight_traj.t0 and self.t <= self.flight_traj.tf:
            return self.flight_traj
        elif self.t > self.mon_traj.t0 and self.t <= self.mon_traj.tf:
            return self.mon_traj
        else:
            # TODO
            err = (f'The current time, {self.t}, is out of bounds of the '
                   f'current trajectories.\n'
                   f'--> Flight: t0={self.flight_traj.t0}, '
                   f'tf={self.flight_traj.tf}\n'
                   f'--> Monitoring: t0={self.mon_traj.t0}, '
                   f'tf={self.mon_traj.tf}')
            raise Exception(err)
#            return self.flight_traj
