#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:29:07 2020

@author: ckielasjensen
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from simulator import Simulator
from target import Target

SIM_TF = 85  # Time at which to stop the simulation (s)
TIME_MULT = 1  # Speed at which virtual time should progress, 1 is real time
TARGET_PERIOD = 1  # Period at which the target is monitored (s)


class Parameters:
    """
    """
    def __init__(self):
        self.deg = 5        # Order of approximation
        self.nveh = 3       # Number of vehicles
        self.ndim = 2       # Number of dimensions
        self.dsafe = 1      # Minimum safe distance between vehicles (m)
        self.odsafe = 2     # Minimum safe distance from obstacles (m)
        self.vmax = 30      # Maximum speed (m/s)
        self.vmin = 1e-12       # Minimum speed (m/s)
        self.wmax = np.pi/4     # Maximum angular rate (rad/s)
#        self.tf = 25.0
        self.tflight = 25.0     # Flight traj time (s)
        self.tmon = 10.0    # Monitoring traj time (s)

        # Target constraints
        self.outerR = 50
        self.innerR = 10
        self.noflyR = 1
        self.detPer = 1     # Detection period of the target (s)

        self.iniPt = np.array([0, 0])
        self.iniSpeed = 3
        self.iniAng = 0
        self.monSpeed = 3
        self.monT = (self.outerR-self.innerR)/self.monSpeed

        # Optimization constraints
        self.degElev = 15

        # Misc
        np.random.seed(3)
        self.iprint = 0     # Verbosity of minimizer output (0, 1, or 2)
        self.relaxation = 1e-3  # Relaxation for final mon point and inner R
        self.replanRad = 5  # If Ept is this different, replan


if __name__ == '__main__':
    # Initialize plots
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.set_xlim(-50, 100)
    ax.set_ylim(-60, 100)

    # Initialize classes
    params = Parameters()
    target = Target(25, 25, 0)
    agents = []
    for i in range(params.nveh):
        agents.append(Agent(0, 15*i, 0, 3, 0, params, ax=ax))

    # Give the target a commanded speed
    target.send_cmd(0.3, 0)

    # Get first plan
    for i, agent in enumerate(agents):
        agent.detect_target(target.get_state())
        agent.compute_flight_traj(dt=(i+1)*params.tflight)
#        agent.compute_mon_traj()
        idxs = list(range(params.nveh))
        idxs.remove(i)
        for j in idxs:
            cpts_f = agent.flight_traj.cpts
            times_f = np.array([agent.flight_traj.t0, agent.flight_traj.tf])
            agents[j].send_traj(cpts_f, times_f)
#            cpts_m = agent.mon_traj.cpts
#            times_m = np.array([agent.mon_traj.t0, agent.mon_traj.tf])
#            agents[j].send_traj(cpts_m, times_m)

    # Plot initial states
    pts = target.get_state()
    trgtPlot = ax.plot(pts[0], pts[1], 'r*', label='Target')
    agentPlots = []
    for i, agent in enumerate(agents):
        pts = agent.get_state()
        agentPlots.append(ax.plot(pts[0], pts[1], 'X', label=f'Agent {i}'))
#        agent.flight_traj.plot(ax, showCpts=False)
#        agent.mon_traj.plot(ax, showCpts=False)
    plt.legend()

    # Run the simulation
#    t0 = time.time()
    t0 = 0.0
    t = 0.0
    t_trgt = 0.0
    agentIdx = 0
    while t < SIM_TF:
        # Detect target
        if t % params.detPer == 0:
            for agent in agents:
                agent.detect_target(target.get_state())

        # Update states
        target.update(t)
        for agent in agents:
            agent.update(t)
            if t - t_trgt >= TARGET_PERIOD:
                t_trgt = t
                agent.detect_target(target.get_state())

        # Update plots
        pts = target.get_state()
        trgtPlot[0].set_data(pts[0], pts[1])
        for i, agent in enumerate(agents):
            pts = agent.get_state()
            agentPlots[i][0].set_data(pts[0], pts[1])
        plt.pause(0.01)

#        t = TIME_MULT*time.time() - t0
        t += 0.1
        if t > 40:
            target.send_cmd(0.3, np.pi/80)
