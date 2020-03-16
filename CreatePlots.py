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

#SIM_TF = 100  # Time at which to stop the simulation (s)
TIME_MULT = 1  # Speed at which virtual time should progress, 1 is real time


class Parameters:
    """
    """
    def __init__(self):
        self.deg = 7        # Order of approximation
        self.nveh = 3       # Number of vehicles
        self.ndim = 2       # Number of dimensions
        self.dsafe = 1      # Minimum safe distance between vehicles (m)
        self.odsafe = 2     # Minimum safe distance from obstacles (m)
        self.vmax = 250      # Maximum speed (m/s)
        self.vmin = 0.1       # Minimum speed (m/s)
        self.wmax = np.pi*10     # Maximum angular rate (rad/s)
#        self.tf = 25.0
        self.tflight = 30.0     # Flight traj time (s)
        self.tmon = 10.0    # Monitoring traj time (s)

        # Target constraints
        self.outerR = 75
        self.innerR = 25
        self.noflyR = 1
        self.detPer = 1     # Detection period of the target (s)

        self.iniPt = np.array([0, 0])
        self.iniSpeed = 3
        self.iniAng = 0
        self.monSpeed = 2
        self.monT = (self.outerR-self.innerR)/self.monSpeed

        # Optimization constraints
        self.degElev = 60

        # Misc
        np.random.seed(0)
        self.iprint = 0     # Verbosity of minimizer output (0, 1, or 2)
        self.relaxation = 1e-3  # Relaxation for final mon point and inner R
        self.replanRad = 5  # If Ept is this different, replan


#if __name__ == '__main__':
def main(ax, SIM_TF):
    # Initialize plots
#    plt.close('all')
#    fig, ax = plt.subplots(1, 1)

    ax.set_aspect('equal')
    ax.set_xlim(-5, 275)
    ax.set_ylim(-50, 125)

    # Initialize classes
    params = Parameters()
    target = Target(25, 25, 0)
    agents = []
    for i in range(params.nveh):
        agents.append(Agent(0, 15*i, 0, 3, 0, params, ax=ax))

    # Give the target a commanded speed
    target.send_cmd(3, 0)

    # Get first plan
    for i, agent in enumerate(agents):
        agent.detect_target(target.get_state())
        agent.compute_flight_traj(tf=params.tflight + i*params.tmon)

    # Plot initial states
    pts = target.get_state()
    trgtPlot = ax.plot(pts[0], pts[1], 'r*', markersize=10, label='Target')
    agentPlots = []
    for i, agent in enumerate(agents):
        pts = agent.get_state()
        agentPlots.append(ax.plot(pts[0], pts[1], f'{Agent.colors[i]}X',
                                  label=f'Agent {i}'))
    ax.legend()

    # Run the simulation
    t0 = 0.0
    t = 0.0
    t_trgt = 0.0
    agentIdx = 0
    while t < SIM_TF:
        # Update states
        target.update(t)
        for agent in agents:
            agent.update(t)
#            if t - t_trgt >= TARGET_PERIOD:
#                t_trgt = t
#                agent.detect_target(target.get_state())

        # Detect target
        if t % params.detPer < 1e-6:
            for agent in agents:
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
#        print(t)
        if t > 43:
            target.send_cmd(3, np.pi/8)

    Agent.agentIdx = 0
    Agent.trajList = []
    Agent.timeList = []
    return

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1)
    main(ax, 100)
