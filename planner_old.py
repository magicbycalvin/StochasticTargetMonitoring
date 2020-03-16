#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:13:19 2020

@author: ckielasjensen
"""

import random
import time

import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import scipy.optimize as sop

import bezier as bez


DEG_ELEV = 30


def flight_plan(params, target, iniPt, iniSpeed, iniAng, trajArray):
    """ Plans the flight trajectory for the current agent
    """
    params.finAng = 2*np.pi*np.random.rand()
    x0 = init_guess(params, target, iniPt, iniSpeed, iniAng)
    def fn(x): return cost_flight(x, target, params)
    cons = [{'type': 'ineq',
             'fun': lambda x: nonlinear_constraints(x, target, trajArray,
                                                    params)}]

    results = sop.minimize(fn, x0,
                           constraints=cons,
                           method='SLSQP',
                           options={'maxiter': 100,
                                    'disp': True,
                                    'iprint': 0})
    if not results.success:
        print(results.message)
    y = reshape(results.x, trajArray, params.nveh, params.deg, params.tf,
                params.iniPt, params.iniSpeed, params.iniAng, params.monSpeed,
                params.finAng)

    return y


def mon_plan():
    """ Plans the monitoring trajectory for the current agent
    """
    pass


def init_guess(params, T, iniPt, iniSpeed, iniAng):
    """Provides an initial guess for the trajectory being planned

    :param params: Object containing the mission parameters
    :type params: Parameters
    :param T: Expected future position of the target
    :type T: np.ndarray
    :param iniPt: Initial point of the vehicle
    :type iniPt: np.ndarray
    :param iniSpeed: Initial speed of the vehicle
    :type iniSpeed: np.ndarray
    :param iniAng: Initial heading of the vehicle
    :type iniAng: np.ndarray
    """
    deg = params.deg

    initMag = iniSpeed*params.tf/deg
    iniX = iniPt[0] + initMag*np.cos(iniAng)       # X
    iniY = iniPt[1] + initMag*np.sin(iniAng)       # Y

    xguess = np.linspace(iniX, T[0], deg)[1:]
    xguess = np.delete(xguess, -2)
    yguess = np.linspace(iniY, T[1], deg)[1:]
    yguess = np.delete(yguess, -2)

    x0 = np.concatenate((xguess, yguess))

    return x0


@njit(cache=True)
def reshape(x, trajArray, nveh, deg, tf, iniPt, iniSpeed, iniAng, finSpeed,
            finAng):
    """Reshapes the X vector being optimized into a matrix for computation

    The optimization vector, X, should be of the following form:
            X = [x2, ... xn-2, xn, y2, ..., yn-2, yn]
            x0 and y0 come from the initial point
            x1 and y1 come from the initial speed and heading
            xn-1 and yn-1 come from the final speed and heading
    The reshaped vector, y, will be a 2xn dimensional array where the rows are
    the spatial dimensions (i.e. X dimension and Y dimension) and the columns
    are the control points for the Bernstein polynomials.

    Note that this only works for 2-dimensional problems.

    :param x: X vector over which the optimization is happening
    :type x: np.ndarray
    :param trajArray: Array of existing trajectories that already match the
        shape of y. These are appended to the created trajectory from X.
    :type trajArray: np.ndarray
    :param nveh: Number of vehicles
    :type nveh: int
    :param deg: Bernstein polynomial degree
    :type deg: int
    :param tf: Final time of the trajectory (assuming t0 = 0)
    :type tf: float
    :param inipt: Initial point of the current vehicle
    :type inipt: np.ndarray
    :param iniSpeed: Initial speed of the current vehicle
    :type iniSpeed: float
    :param iniAng: Initial heading of the current vehicle, in radians
    :type iniAng: float
    :param finSpeed: Final speed of the current vehicle
    :type finSpeed: float
    :param finAng: Final angle of the current vehicle
    :type finAng: float
    :return: Reshaped vector, y
    :rtype: np.ndarray
    """
    ndim = 2

    # Reshape X
    y = np.empty((ndim*nveh, deg+1))
    reshapedX = x.reshape((ndim, -1))
    y[:ndim, 2:-2] = reshapedX[:, :-1]

    # Initial points
    y[0, 0] = iniPt[0]
    y[1, 0] = iniPt[1]
    y[:ndim, -1] = reshapedX[:, -1]

    # Initial and final speeds and headings
    initMag = iniSpeed*tf/deg
    y[0, 1] = iniPt[0] + initMag*np.cos(iniAng)     # X
    y[1, 1] = iniPt[1] + initMag*np.sin(iniAng)     # Y
    finMag = finSpeed*tf/deg
    y[0, -2] = reshapedX[0, -1] + finMag*np.cos(finAng)
    y[1, -2] = reshapedX[1, -1] + finMag*np.sin(finAng)

    # Add in the rest of the trajectories
    if trajArray.size > 0:
        y[ndim:, :] = trajArray

    return y


def build_traj_list(y, tf, params):
    """
    """
    trajs = []
    for i in range(params.nveh):
        t0 = params.nveh - i
        trajs.append(bez.Bezier(y[i*params.ndim:(i+1)*params.ndim, :],
                                t0=t0, tf=t0+tf))

    return trajs


def cost_flight(x, target, params):
    """Cost function for the flight trajectory

    To grab the proper index value, we use the following formulas:
        xidx = 1*(params.deg-3) + 0
        yidx = 2*(params.deg-3) + 1
        zidx = 3*(params.deg-3) + 2
    where each dimension we increase (x, y, z, etc.) is an increasing multiple.
    We also have to add 1 for each increased dimension since the length of
    the polynomials is deg+1. The -3 comes from the predefined values of the
    initial point, speed, and heading, and the final speed and heading.
    """
    xidx = params.deg - 3
    yidx = 2*params.deg - 5
    xpos = x[xidx]
    ypos = x[yidx]
    trgtX = target[0] + params.outerR*np.cos(params.finAng)
    trgtY = target[1] + params.outerR*np.sin(params.finAng)
    val = np.linalg.norm([trgtX - xpos,
                          trgtY - ypos])

    return val


def nonlinear_constraints(x, target, trajArray, params):
    """
    """
    y = reshape(x, trajArray, params.nveh, params.deg, params.tf,
                params.iniPt, params.iniSpeed, params.iniAng, params.monSpeed,
                params.finAng)
    trajs = build_traj_list(y, params.tf, params)

    nonlcon = np.concatenate([temporal_separation_cons(trajs, params),
                              max_speed_cons(trajs[0], params),
                              max_ang_rate_cons(trajs[0], params)
                              ])

    return nonlcon


def temporal_separation_cons(trajs, params):
    """
    """
#    nveh = params.nveh
    nveh = len(trajs)

    if nveh > 1:
        traj = trajs[0]
        distVeh = []
        for i, veh in enumerate(trajs[1:]):
            dv = traj - veh
            if dv is not None:
                distVeh.append(dv.normSquare().elev(DEG_ELEV).cpts.squeeze())

        return np.concatenate(distVeh) - params.dsafe**2

    else:
        return np.atleast_1d(0.0)


def max_speed_cons(traj, params):
    """Creates the maximum velocity constraints.

    Useful for limiting the maximum speed of a vehicle.

    :param maxSpeed: Maximum speed of the vehicle.
    :type maxSpeed: float
    :return: Inequality constraint for the maximum speed
    :rtype: float
    """
    speedSqr = traj.diff().normSquare().elev(DEG_ELEV).cpts.squeeze()

    return params.vmax**2 - speedSqr


def max_ang_rate_cons(traj, params):
    """
    """
    angRateSqr = angular_rate_sqr(traj.elev(DEG_ELEV)).cpts.squeeze()

    return params.wmax**2 - angRateSqr


def angular_rate_sqr(traj):
    """
    Finds the squared angular rate of the 2D Bezier Curve.

    The equation for the angular rate is as follows:
        psiDot = ((yDdot*xDot - xDdot*yDot))^2 / (xDot^2 + yDot^2)^2
        Note the second derivative (Ddot) vs the first (Dot)

    RETURNS:
        RationalBezier - This function returns a rational Bezier curve because
            we must divide two Bezier curves.
    """
    x = traj.x
    xDot = x.diff()
    xDdot = xDot.diff()

    y = traj.y
    yDot = y.diff()
    yDdot = yDot.diff()

    numerator = yDdot*xDot - xDdot*yDot
    numerator = numerator*numerator
    denominator = xDot*xDot + yDot*yDot
    denominator = denominator*denominator

    cpts = np.nan_to_num(numerator.cpts / (denominator.cpts))
    weights = denominator.cpts

    return bez.RationalBezier(cpts, weights)
