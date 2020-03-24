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
from parameters import Parameters
from target import Target


XLIM = (-50, 350)
YLIM = (-100, 110)


def plot1():
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)

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
    for i, agent in enumerate(agents):
        agent.plot_arrow()

    # Plot the inner and outer radii
    cir1 = Circle(target.get_state()[:2], ls=':', fill=False, ec='r',
                  label='Outer Radius', radius=params.outerR)
    cir2 = Circle(target.get_state()[:2], ls=':', fill=False, ec='r',
                  label='Outer Radius', radius=params.innerR)
    cir3 = Circle(target.get_state()[:2], lw=None, fc='gray', alpha=0.5,
                  label='No Fly Zone', radius=params.noflyR)
    ax.add_artist(cir1)
    ax.add_artist(cir2)
    ax.add_artist(cir3)

    # Draw legend and clean up Agent class
    ax.legend([trgtPlot[0]] + [agent._arrow for agent in agents],
              ['Target',
               'Agent 1',
               'Agent 2',
               'Agent 3'])
    Agent.agentIdx = 0
    Agent.trajList = []
    Agent.timeList = []

    plt.title('$t = 0.0 s$')
    return ax, target, agents


def plot2(tb=60.0, targetTracer=False):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
#    mng = plt.get_current_fig_manager()
#    mng.resize(1920, 1080)

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
        agent.plot_arrow()

    # Plot initial states
    pts = target.get_state()
    trgtPlot = ax.plot(pts[0], pts[1], 'r*', markersize=10, label='Target')
    for i, agent in enumerate(agents):
        agent.plot_arrow()

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
            agent.plot_arrow()

        if t >= 1:
            target.send_cmd(3, 0)

        if targetTracer:
            if np.abs(t-30) < 1e-4:
                temp = target.get_state()
                ax.plot(temp[0], temp[1], 'r*', markersize=13,
                        fillstyle='none')
                ax.text(temp[0]-10, temp[1]+3, '$t = 30s$', fontsize=12)

            elif np.abs(t-40) < 1e-4:
                temp = target.get_state()
                ax.plot(temp[0], temp[1], 'r*', markersize=13,
                        fillstyle='none')
                ax.text(temp[0]-10, temp[1]+3, '$t = 40s$', fontsize=12)

        if t >= tb:
            break

#        if t >= 5:
#            target.send_cmd(8, np.pi/2)
#
#        if t >= 5.5:
#            target.send_cmd(8, 0)

        plt.pause(0.01)
#        if t >= 20:
#            break

    # Plot the inner and outer radii
    cir1 = Circle(target.get_state()[:2], ls=':', fill=False, ec='r',
                  label='Outer Radius', radius=params.outerR)
    cir2 = Circle(target.get_state()[:2], ls=':', fill=False, ec='r',
                  label='Inner Radius', radius=params.innerR)
    cir3 = Circle(target.get_state()[:2], lw=None, fc='gray', alpha=0.5,
                  label='No Fly Zone', radius=params.noflyR)
    ax.add_artist(cir1)
    ax.add_artist(cir2)
    ax.add_artist(cir3)

    # Draw legend and clean up Agent class
    ax.legend([trgtPlot[0]] + [agent._arrow for agent in agents],
              ['Target',
               'Agent 1',
               'Agent 2',
               'Agent 3'])
    Agent.agentIdx = 0
    Agent.trajList = []
    Agent.timeList = []

    plt.title(f'$t = {t} s$')
    return ax, target, agents


def main(p2tb=60.0, targetTracer=False):
    p2ret = plot2(tb=p2tb, targetTracer=targetTracer)
    p1ret = plot1()

    return p1ret, p2ret


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
    p1ret, p2ret = main()
