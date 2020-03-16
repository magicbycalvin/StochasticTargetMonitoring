#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:43:20 2020

@author: ckielasjensen
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from simulator import Simulator
from target import Target

SIM_TF = 120  # Time at which to stop the simulation (s)
TIME_MULT = 1  # Speed at which virtual time should progress, 1 is real time
TARGET_PERIOD = 1  # Period at which the target is monitored (s)


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
        self.wmax = np.pi/2     # Maximum angular rate (rad/s)
#        self.tf = 25.0
        self.tflight = 25.0     # Flight traj time (s)
        self.tmon = 10.0    # Monitoring traj time (s)

        # Target constraints
        self.outerR = 5
        self.innerR = 3
        self.noflyR = 1

        self.iniPt = np.array([0, 0])
        self.iniSpeed = 3
        self.iniAng = 0
        self.monSpeed = 3
        self.monT = (self.outerR-self.innerR)/self.monSpeed

        # Optimization constraints
        self.degElev = 30

        # Misc
        self.randSeed = 3   # Seed for numpy's RNG
        self.iprint = 2     # Verbosity of minimizer output (0, 1, or 2)


if __name__ == '__main__':
    # Initialize classes
    params = Parameters()
    target = Target(25, 25, 0)
    agents = []
    for i in range(params.nveh):
        agents.append(Agent(0, 10*i, 0, 0, 0, params))

    # Get first plan
    for agent in agents:
        agent.detect_target(target.get_state())
        agent.plan_flight()
        agent.plan_mon()

    # Initialize plots
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)

    # Plot initial states
    pts = target.get_state()
    trgtPlot = ax.plot(pts[0], pts[1], 'r*', label='Target')
    agentPlots = []
    for i, agent in enumerate(agents):
        pts = agent.get_state()
        agentPlots.append(ax.plot(pts[0], pts[1], 'X', label=f'Agent {i}'))
    plt.legend()

    # Give the target a commanded speed
    target.send_cmd(0.5, 0)

    # Run the simulation
#    t0 = time.time()
    t0 = 0.0
    t = 0.0
    t_trgt = 0.0
    while t < SIM_TF:
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
            target.send_cmd(0.5, np.pi/40)
