#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 06:18:59 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from agent import Agent
from target import Target


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
        self.outerR = 75
        self.innerR = 25
        self.noflyR = 1
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


def plot1():
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-60, 110)
    ax.set_ylim(-60, 110)

    # Initialize classes
    params = Parameters()
    target = Target(25, 25, 0)
    agents = []
    for i in range(params.nveh):
        agents.append(Agent(0, 25*i, 0, params.monSpeed, 0, params, ax=ax))

    # Give the target a commanded speed
    target.send_cmd(0, 0)

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

    # Plot the inner and outer radii
    cir1 = Circle(target.get_state()[:2], ls=':', fill=False, ec='r',
                  label='Outer Radius', radius=params.outerR)
    cir2 = Circle(target.get_state()[:2], ls=':', fill=False, ec='r',
                  label='Outer Radius', radius=params.innerR)
    ax.add_artist(cir1)
    ax.add_artist(cir2)

    # Draw legend and clean up Agent class
    ax.legend()
    Agent.agentIdx = 0
    Agent.trajList = []
    Agent.timeList = []

    plt.title('$t = 0$')
    return


def plot2():
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-60, 110)
    ax.set_ylim(-60, 110)

    # Initialize classes
    params = Parameters()
    target = Target(25, 25, 0)
    agents = []
    for i in range(params.nveh):
        agents.append(Agent(0, 25*i, 0, params.monSpeed, 0, params, ax=ax))

    # Give the target a commanded speed
    target.send_cmd(0, 0)

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

    # Run the simulation
    for t in np.arange(0, params.tflight + params.nveh*params.tmon + 0.1, 0.1):
        # Update states
        target.update(t)
        for agent in agents:
            agent.update(t)

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

    # Plot the inner and outer radii
    cir1 = Circle(target.get_state()[:2], ls=':', fill=False, ec='r',
                  label='Outer Radius', radius=params.outerR)
    cir2 = Circle(target.get_state()[:2], ls=':', fill=False, ec='r',
                  label='Outer Radius', radius=params.innerR)
    ax.add_artist(cir1)
    ax.add_artist(cir2)

    ax.legend()
    Agent.agentIdx = 0
    Agent.trajList = []
    Agent.timeList = []

    plt.title(f'$t = {t}$')
    return


def main():
    plot1()
    plot2()


if __name__ == '__main__':
    plt.rcParams.update({
            'font.size': 20,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'lines.linewidth': 2,
            'lines.markersize': 9
            })

#    plt.rcParams.update(plt.rcParamsDefault)
#    plt.ion()
    plt.close('all')
    main()
