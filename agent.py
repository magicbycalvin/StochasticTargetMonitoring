#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:37:50 2020

@author: ckielasjensen
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize

from bezier import Bezier
from planner_new import plan_flight, plan_mon


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
        self._traj_list = np.atleast_2d([])
        self._time_list = np.atleast_2d([])
        self._traj_state = 'flight'
        self._ax = ax
        self._last_trgt = None

    # TODO
    #   * Fix the dt=None since it is a messy workaround for adding differently
    #     timed trajectories
    def compute_flight_traj(self, dt=None):
        """ Plans the flight trajectory
        """
        print(f'Agent {self.idx} computing flight')
        p0 = self.state[:2]
        v0 = self.state[3]
        psi0 = self.state[2]
        t0 = self.t
        if dt is None:
            dt = self.params.tflight
        trgt = self.predict_target(t0 + dt)
        self._last_trgt = trgt.copy()
        pastCpts = self._traj_list
        pastTimes = self._time_list
        trgt_cpts = self.predict_trgt_traj(t0+dt)
        flight_traj = plan_flight(p0, v0, psi0, t0, trgt, trgt_cpts, pastCpts,
                                  pastTimes, self.params, dt=dt)
        self.flight_traj = flight_traj

        if self._ax is not None:
            flight_traj.plot(self._ax, showCpts=False)
            plt.pause(0.001)

    def compute_mon_traj(self, dt=None):
        """ Plans the monitoring trajectory
        """
        print(f'Agent {self.idx} computing mon')
        pdot = self.flight_traj.diff()
        p0 = self.flight_traj.cpts[:, -1]
        v0 = np.linalg.norm(pdot.cpts[:, -1])
        psi0 = np.arctan2(pdot.cpts[1, -1], pdot.cpts[0, -1])
        t0 = self.flight_traj.tf
        pastCpts = self._traj_list
        pastTimes = self._time_list
        if dt is None:
            dt = self.params.tmon
        trgt_cpts = self.predict_trgt_traj(t0+dt)

        mon_traj = plan_mon(p0, v0, psi0, t0, trgt_cpts, pastCpts, pastTimes,
                            dt, self.params)

        self.mon_traj = mon_traj

        if self._ax is not None:
            mon_traj.plot(self._ax, showCpts=False)
            plt.pause(0.001)

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
            self._traj_state = 'flight'
            return self.flight_traj
        elif self.t > self.mon_traj.t0 and self.t <= self.mon_traj.tf:
            self._traj_state = 'mon'
            return self.mon_traj
        else:
            # TODO
            err = (f'The current time, {self.t}, is out of bounds of the '
                   f'current trajectories.\n'
                   f'--> Flight: t0={self.flight_traj.t0}, '
                   f'tf={self.flight_traj.tf}\n'
                   f'--> Monitoring: t0={self.mon_traj.t0}, '
                   f'tf={self.mon_traj.tf}')
#            raise Exception(err)
            return self.flight_traj

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
                print('======= REPLANNING! =======')
                print(f'tf: {self.flight_traj.tf}, t: {self.t}')
                self.compute_flight_traj(dt=self.flight_traj.tf - self.t)

    def predict_target(self, tf):
        """Predicts the future location of the target

        Ept - Expectation of the target position, pt
        """
        y0 = self.trgt_state[:3]
        v = self.trgt_state[3]
        w = self.trgt_state[4]

        def fn(t, x): return [v*np.cos(x[2]), v*np.sin(x[2]), w]

        sol = solve_ivp(fn, (0, tf), y0)

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

        sol = solve_ivp(fn, (0, tf), y0, t_eval=np.linspace(0, tf, npts))

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

#    def send_trajectories(self, idx, trajectories):
#        """ Sends one agent's trajectories to the current agent
#
#        :param idx: Index of the agent whose trajectories are being sent
#        :type idx: int
#        :param trajectories: Dict containing the monitoring trajectory and
#            flight trajectory of the agent. The trajectories should be Bezier
#            objects and the dictionary entries should be 'mon_traj' and
#            'flight_traj'
#        :type trajectories: dict
#        """
#        self._traj_list[idx] = trajectories.copy()

    def send_traj(self, traj, times):
        """ Sends an existing agent's trajectory to the current agent

        :param traj: Matrix of control points defining the trajectory where
            each row corresponds to the dimension and the columns correspond
            to the control points
        :type traj: np.ndarray
        :param times: Vector containing t0 and tf, [t0, tf]
        :type times: np.ndarray
        """
        if self._traj_list.shape[1] > 0:
            self._traj_list = np.vstack((self._traj_list, traj))
        else:
            self._traj_list = traj.copy()

        if self._time_list.shape[1] > 0:
            self._time_list = np.vstack((self._time_list, times))
        else:
            self._time_list = np.atleast_2d(times.copy())

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
        self.state[3] = trajDotNormSqr
        if trajDotNormSqr == 0:
            self.state[0] = 0.0
        else:
            self.state[4] = ((xdot(t)*yddot(t) - xddot(t)*ydot(t)) /
                             trajDotNormSqr)

    def replan_check(self):
        """
        """
        if self._traj_state == 'mon':
            if self.t >= self.mon_traj.tf - 1e-9:
                self.compute_flight_traj()

        elif self._traj_state == 'flight':
            if self.t >= self.flight_traj.tf - 1e-9:
                self.compute_mon_traj()
