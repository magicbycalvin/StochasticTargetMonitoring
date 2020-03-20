#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:43:41 2020

@author: ckielasjensen
"""
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from scipy.optimize import minimize, Bounds

from bezier import Bezier, RationalBezier


def plan_flight(p0, v0, psi0, t0, trgt, trgt_cpts, pastCpts, pastTimes, params,
                tf=None):
    """
    """
    randomizer = 0
    while True:
        ndim = 2
        vf = params.monSpeed
        psif = 2*np.pi*np.random.rand()
        if tf is None:
            tf = t0 + params.tflight
        dt = tf - t0
        assert dt >= 0, f'dt should be >= 0, t0: {t0}, tf: {tf}'

        nveh = pastCpts.shape[0] // ndim
        trgt_traj = Bezier(trgt_cpts, t0=t0, tf=tf)

        bounds = Bounds(min(p0) - 250, max(p0) + 250)
        x0 = init_guess_f(p0, trgt, v0, vf, psi0, psif, dt, params.deg)
        x0 += np.random.randn()*randomizer
        def fn(x): return cost_f(x, trgt, p0, v0, vf, psi0, psif, dt,
                                 params.deg, params)
        cons = [{'type': 'ineq',
                 'fun': lambda x: nonlinear_constraints_f(x, p0, v0, vf, psi0,
                                                          psif, t0, tf, nveh,
                                                          trgt_traj,
                                                          pastCpts, pastTimes,
                                                          params)}]

        results = minimize(fn, x0,
                           constraints=cons,
                           bounds=bounds,
                           method='SLSQP',
                           options={'maxiter': 250,
                                    'disp': True,
                                    'iprint': params.iprint})

        if not results.success:
            print(results.message)
            print(f'ETarget: {trgt}')
            print(f'Psif: {psif}')
            print(f'Cost: {fn(results.x)}')
            print(f'Nonlcon:')
            temp = cons[0]['fun'](results.x)
            names = iter(['sep', 'max', 'min', 'max ang', 'NFZ'])
            print(next(names))
            for val in temp:
                if val == 99999:
                    print('###')
                    print(next(names))
                else:
                    print(np.round(val, 3), end=', ')
            cost_f(results.x, trgt, p0, v0, vf, psi0, psif, dt, params.deg,
                   params,
                   disp=True)
            print()
            randomizer += 5
#            if randomizer > 10:
#                raise Exception('Maximum iterations met!')
#
            y = reshape_f(results.x, p0, v0, vf, psi0, psif, dt, params.deg)
            newTraj = Bezier(y, t0=t0, tf=tf)
#
#            newTraj.plot()
#            plt.title('Trajectory')
#            newTraj.diff().normSquare().elev(params.degElev).plot()
#            plt.title('Norm Square')
#            0/0
            if min(cons[0]['fun'](results.x)) < -100000:
                newTraj.diff().normSquare().elev(params.degElev).plot()
                plt.title('Norm Square')
                0/0

        y = reshape_f(results.x, p0, v0, vf, psi0, psif, dt, params.deg)
        newTraj = Bezier(y, t0=t0, tf=tf)

        if results.success:
            break
        elif randomizer > 15:
            break
        break

    return newTraj


def plan_mon(p0, v0, psi0, t0, trgt_cpts, pastCpts, pastTimes, tf, params):
    """

    MON CONSTRAINTS:
        * Usual (speed, rate, Ds)
        * Must be somewhere along the inner monitoring radius
    """
    ndim = 2
    vf = params.monSpeed
    nveh = pastCpts.shape[0] // ndim
    trgt = trgt_cpts[:, -1]
    trgt_traj = Bezier(trgt_cpts, t0=t0, tf=tf)

    dt = tf - t0
    assert dt >= 0, f'dt should be >= 0, t0: {t0}, tf: {tf}'

    x0 = init_guess_m(p0, trgt, v0, vf, psi0, dt, params.deg)
    def fn(x): return cost_m(x, trgt_cpts, p0, v0, psi0, dt, params)
    cons = [{'type': 'ineq',
             'fun': lambda x: nonlinear_constraints_m(x, p0, v0, vf, psi0,
                                                      t0, tf, nveh, pastCpts,
                                                      pastTimes, trgt,
                                                      trgt_traj, params)}]

    results = minimize(fn, x0,
                       constraints=cons,
                       method='SLSQP',
                       options={'maxiter': 250,
                                'disp': True,
                                'iprint': params.iprint})

    y = reshape_m(results.x, p0, v0, psi0, dt, params.deg)
    newTraj = Bezier(y, t0=t0, tf=tf)

    return newTraj


# TODO
#   * Make initial guess better by setting the final point on the outer
#     monitoring radius instead of right on top of the target
def init_guess_f(p0, trgt, v0, vf, psi0, psif, dt, deg):
    """Straight line initial guess for the optimizer

    :param p0: Initial 2D position of the agent, (x, y)
    :type p0: np.ndarray
    :param v0: Initial speed
    :type v0: float
    :param vf: Final speed
    :type vf: float
    :param psi0: Initial heading
    :type psi0: float
    :param psif: Final heading
    :type psif: float
    :param dt: Difference between final and initial time (i.e. tf - t0), used
        for finding the magnitude of the vector between the first two control
        points and between the last two control points
    :type dt: float
    :param deg: Degree of the Bernstein polynomials being used
    :type deg: int
    :return: Initial guess for the optimizer
    :rtype: np.ndarray
    """
    initMag = v0*dt/deg
    x1 = p0[0] + initMag*np.cos(psi0)
    y1 = p0[1] + initMag*np.sin(psi0)

    finalMag = vf*dt/deg
    xn_1 = trgt[0] - finalMag*np.cos(psif)
    yn_1 = trgt[1] - finalMag*np.sin(psif)

    xguess = np.linspace(x1, xn_1, deg-1)[1:-1]
    xguess = np.append(xguess, trgt[0])
    yguess = np.linspace(y1, yn_1, deg-1)[1:-1]
    yguess = np.append(yguess, trgt[1])

    x0 = np.concatenate((xguess, yguess))

    return x0


def init_guess_m(p0, trgt, v0, vf, psi0, dt, deg):
    """
    """
    initMag = v0*dt/deg
    x1 = p0[0] + initMag*np.cos(psi0)
    y1 = p0[1] + initMag*np.sin(psi0)

    psif = np.arctan2(trgt[1]-y1, trgt[0]-x1)
    finalMag = vf*dt/deg
    xn = trgt[0] - finalMag*np.cos(psif)
    yn = trgt[1] - finalMag*np.sin(psif)

    xguess = np.linspace(x1, xn, deg)[1:]
    yguess = np.linspace(y1, yn, deg)[1:]

    x0 = np.concatenate((xguess, yguess))

    return x0


def cost_f(x, trgt, p0, v0, vf, psi0, psif, dt, deg, params, disp=False):
    """Cost function for the flight trajectory

    The cost is defined as the straight line distance between the final control
    point of the agent's trajectory and some random point along the outer
    monitoring radius around the expected position of the target.

    Note to devs:
    To grab the proper index value, we use the following formulas:
        xidx = 1*(params.deg-3) + 0
        yidx = 2*(params.deg-3) + 1
        zidx = 3*(params.deg-3) + 2
    where each dimension we increase (x, y, z, etc.) is an increasing multiple.
    We also have to add 1 for each increased dimension since the length of
    the polynomials is deg+1. The -3 comes from the predefined values of the
    initial point, speed, and heading, and the final speed and heading.

    :param x: Optimization vector
    :type x: np.ndarray
    :param trgt: Expected 2D position of target at tf, (x, y)
    :type trgt: np.ndarray
    :param psif: Final heading angle
    :type psif: float
    :param params: Object containing the mission parameters
    :type params: Parameters
    :return: Cost of the current optimization iteration
    :rtype: float
    """
    y = reshape_f(x, p0, v0, vf, psi0, psif, dt, deg)
#    xidx = params.deg - 3
#    yidx = 2*params.deg - 5
#    xpos = x[xidx]
#    ypos = x[yidx]
#    assert xpos == y[0, -1], 'xpos'
#    assert ypos == y[1, -1], 'ypos'
    xpos = y[0, -1]
    ypos = y[1, -1]
    # Adding pi to psif since the heading angle points directly opposite the
    # direction of the vector from the target to the random point along the
    # outer monitoring radius
    trgtX = trgt[0] + params.outerR*np.cos(psif+np.pi)
    trgtY = trgt[1] + params.outerR*np.sin(psif+np.pi)
    finalPosCost = np.linalg.norm([trgtX - xpos,
                                   trgtY - ypos])

    # Min euclidean distance between cpts
    euclidCost = _euclideanObjective(y, 1, 2)

    if disp:
        print()
        print(f'Pos Cost: {finalPosCost}')
        print(f'Euclid Cost: {euclidCost}')

    return 100*finalPosCost + euclidCost


# TODO
#   * Since the target trajectory object is created here and in the main
#     function, pass it in here rather than create it in the cost each time
def cost_m(x, trgt_cpts, p0, v0, psi0, dt, params):
    """Cost function for the monitoring trajectory
    """
    pt = Bezier(trgt_cpts, tf=dt)
    y = reshape_m(x, p0, v0, psi0, dt, params.deg)
    p = Bezier(y, tf=dt).elev(params.degElev)
    pdot = p.diff()

    costPts = ((pt.y - p.y)*pdot.x - (pt.x - p.x)*pdot.y).normSquare().cpts

#    if np.sign(sum(pdot*(pt-p)))

    return sum(costPts.squeeze())


@njit(cache=True)
def reshape_f(x, p0, v0, vf, psi0, psif, dt, deg):
    """Reshapes the optimization vector x into a usable matrix y

    :param x: Optimization vector being reshaped
    :type x: np.ndarray
    :param p0: Initial 2D position of the agent, (x, y)
    :type p0: np.ndarray
    :param v0: Initial speed
    :type v0: float
    :param vf: Final speed
    :type vf: float
    :param psi0: Initial heading
    :type psi0: float
    :param psif: Final heading
    :type psif: float
    :param dt: Difference between final and initial time (i.e. tf - t0), used
        for finding the magnitude of the vector between the first two control
        points and between the last two control points
    :type dt: float
    :param deg: Degree of the Bernstein polynomials being used
    :type deg: int
    """
    ndim = 2

    # Reshape X
    y = np.empty((ndim, deg+1))
    reshapedX = x.reshape((ndim, -1))
    y[:, 2:-2] = reshapedX[:, :-1]

    # Initial and final points
    y[0, 0] = p0[0]
    y[1, 0] = p0[1]
    y[:, -1] = reshapedX[:, -1]

    # Initial and final speeds and headings
    initMag = v0*dt/deg
    y[0, 1] = p0[0] + initMag*np.cos(psi0)
    y[1, 1] = p0[1] + initMag*np.sin(psi0)
    finMag = vf*dt/deg
    y[0, -2] = reshapedX[0, -1] - finMag*np.cos(psif)
    y[1, -2] = reshapedX[1, -1] - finMag*np.sin(psif)

    return y


@njit(cache=True)
def reshape_m(x, p0, v0, psi0, dt, deg):
    """

    NO FINAL PSI OR SPEED
    """
    ndim = 2

    y = np.empty((ndim, deg+1))
    reshapedX = x.reshape((ndim, -1))
    y[:, 2:] = reshapedX

    y[0, 0] = p0[0]
    y[1, 0] = p0[1]

    initMag = v0*dt/deg
    y[0, 1] = p0[0] + initMag*np.cos(psi0)
    y[1, 1] = p0[1] + initMag*np.sin(psi0)

    return y


def build_traj_list(cpts, times, ndim, nveh):
    """Builds a trajectory list of Bernstein polynomial objects

    :param cpts: Control points of the polynomials where the first m rows are
        each dimension of the first trajectory. Following sets of m rows
        correspond to each trajectory after the first one. The columns hold
        each control point for the trajectories. There should be ndim*nveh
        rows and deg+1 columns.
    :type cpts: np.ndarray
    :param times: Array of initial and final times for each trajectory. Each
        row corresponds to each trajectory. The first column is t0 and the
        second column is tf.
    :type times: np.ndarray
    :param ndim: Number of dimensions (e.g. 2D, 3D, etc.). Most likely 2 or 3.
    :type ndim: int
    :param nveh: Number of trajectories (vehicles)
    :type nveh: int
    :return: List of Bernstein polynomial objects (Bezier) corresponding to
        each trajectory passed in.
    :rtype: list(Bezier)
    """
    trajs = []
    for i in range(nveh):
        trajs.append(Bezier(cpts[i*ndim:(i+1)*ndim, :],
                            t0=times[i, 0],
                            tf=times[i, 1]))

    return trajs


def nonlinear_constraints_f(x, p0, v0, vf, psi0, psif, t0, tf,
                            nveh, trgt_traj, pastCpts, pastTimes, params):
    """Nonlinear constraints for the optimization problem
    """
    ndim = 2

    dt = tf - t0
    y = reshape_f(x, p0, v0, vf, psi0, psif, dt, params.deg)
    if nveh > 0:
        cpts = np.vstack((y, pastCpts))
        times = np.vstack(([t0, tf], pastTimes))
    else:
        cpts = y
        times = np.atleast_2d([t0, tf])

    trajs = build_traj_list(cpts, times, ndim, nveh+1)

    nonlcon = np.concatenate([temporal_sep_con(trajs, nveh, params),
                              [99999],
                              max_speed_con(trajs[0], params),
                              [99999],
                              min_speed_con(trajs[0], params),
                              [99999],
                              max_angrate_con(trajs[0], params),
                              [99999],
                              noflyzone_con(trajs[0], trgt_traj, params)
                              ])

    return nonlcon


# TODO
#   * remove trgt and just have trgt_traj
def nonlinear_constraints_m(x, p0, v0, vf, psi0, t0, tf,
                            nveh, pastCpts, pastTimes, trgt, trgt_traj,
                            params):
    """Nonlinear constraints for the optimization problem
    """
    ndim = 2

    dt = tf - t0
    y = reshape_m(x, p0, v0, psi0, dt, params.deg)
    if nveh > 0:
        cpts = np.vstack((y, pastCpts))
        times = np.vstack(([t0, tf], pastTimes))
    else:
        cpts = y
        times = np.atleast_2d([t0, tf])

    trajs = build_traj_list(cpts, times, ndim, nveh+1)

    nonlcon = np.concatenate([temporal_sep_con(trajs, nveh, params),
                              max_speed_con(trajs[0], params),
                              min_speed_con(trajs[0], params),
                              max_angrate_con(trajs[0], params),
                              final_pos_con(trajs[0], trgt, params),
                              noflyzone_con(trajs[0], trgt_traj, params)
                              ])

    return nonlcon


def temporal_sep_con(trajs, nveh, params):
    """
    """
    if nveh > 1:
        traj = trajs[0]
        distVeh = []
        for i, veh in enumerate(trajs[1:]):
            dv = traj - veh
            if dv is not None:
                distVeh.append(
                        dv.normSquare().elev(params.degElev).cpts.squeeze())

        # If no trajectories match in time, we don't need collision checking
        if len(distVeh) == 0:
            return np.atleast_1d(0.0)

        return np.concatenate(distVeh) - params.dsafe**2

    else:
        return np.atleast_1d(0.0)


# TODO
#   * Integrade the constraints so that the derivatives and degree elevations
#     are only computed once rather than each function call
def max_speed_con(traj, params):
    """Computes the maximum speed constraints

    Used for limiting the maximum speed of a vehicle.

    :param traj: Bernstein polynomial (Bezier) object of the position of the
        vehicle
    :type traj: Bezier
    :param params: Mission parameters
    :type params: Parameters
    :return: Inequality constraint for the maximum speed
    :rtype: float
    """
    speedSqr = traj.diff().normSquare().elev(params.degElev).cpts.squeeze()

    return params.vmax**2 - speedSqr


def min_speed_con(traj, params):
    """Computes the minimum speed constraints

    Used for limiting the minimum speed of a vehicle.

    :param traj: Bernstein polynomial (Bezier) object of the position of the
        vehicle
    :type traj: Bezier
    :param params: Mission parameters
    :type params: Parameters
    :return: Inequality constraint for the minimum speed
    :rtype: float
    """
#    temp = traj.diff().normSquare().elev(params.degElev)
#    i = 1
#    while np.any(temp.cpts.squeeze() < 0):
#        print(f'Deg Elev: {i}')
#        temp = temp.elev(10)
#    speedSqr = temp.cpts.squeeze()
    speedSqr = traj.diff().normSquare().elev(params.degElev).cpts.squeeze()

    return speedSqr - params.vmin**2


def max_angrate_con(traj, params):
    """
    """
    angRateSqr = angular_rate_sqr(traj.elev(params.degElev)).cpts.squeeze()

    return params.wmax**2 - angRateSqr


def noflyzone_con(traj, trgt_traj, params):
    """
    """
    p = traj.elev(params.degElev)
    try:
        dv = (p - trgt_traj).normSquare().cpts.squeeze()
    except Exception as e:
        print(p)
        print(trgt_traj)
        raise(e)

    return dv - params.noflyR**2


def final_pos_con(traj, trgt, params):
    """
    """
    dist = np.linalg.norm(traj.cpts[:, -1] - trgt)

    return [params.relaxation - np.abs(params.innerR - dist)]


def angular_rate_sqr(traj):
    """Finds the squared angular rate of a 2D Bezier Curve

    The equation for the angular rate is as follows:
        psiDot = ((yDdot*xDot - xDdot*yDot))^2 / (xDot^2 + yDot^2)^2
        Note the second derivative (Ddot) vs the first (Dot)

    NOTE: If this function is causing issues in the optimization, it is likely
    due to an initial speed of zero causing weirdness in the angular rate
    calculation

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
#    cpts[np.abs(cpts) < 1e-9] = 0.0  # Added to get rid of very small values
    weights = denominator.cpts

    return RationalBezier(cpts, weights)


# TODO Figure out why this returns such large nuumbers (likely due to variable
#   type issues with numba)
#@njit(cache=True)
def _euclideanObjective(y, nVeh, dim):
    """Sums the Euclidean distance between control points.

    The Euclidean difference between each neighboring pair of control points is
    summed for each vehicle.

    :param y: Optimized vector that has been reshaped using the reshapeVector
        function.
    :type y: numpy.ndarray
    :param nVeh: Number of vehicles
    :type nVeh: int
    :param dim: Dimension of the vehicles. Currently only works for 2D
    :type dim: int
    :return: Sum of the Euclidean distances
    :rtype: float
    """
    summation = 0.0
    temp = np.empty(3)
    length = y.shape[1]
    for veh in range(nVeh):
        for i in range(length-1):
            for j in range(dim):
                temp[j] = y[veh*dim+j, i+1] - y[veh*dim+j, i]

            summation += _norm(temp)  # np.linalg.norm(temp)

    return summation


#@njit(cache=True)
def _norm(x):
    """
    """
    summation = 0.
    for val in x:
        summation += val*val

    return np.sqrt(summation)


if __name__ == '__main__':
    deg = 5
    dt = 10.0
    p0 = np.array([0, 3], dtype=float)
    v0 = 0.5
    vf = 0.5
    psi0 = 0.0
    psif = 0.0
    trgt = np.array([5, 9], dtype=float)

    print('Testing reshape_f')
    ytrue = np.array([[0, 1, 2, 3, 4, 5],
                      [3, 3, 6, 2, 9, 9]], dtype=float)
    x = np.array([2, 3, 5, 6, 2, 9])
    y = reshape_f(x, p0, v0, vf, psi0, psif, dt, deg)
    if not np.all(y == ytrue):
        print('--> [!] Test failed')
    else:
        print('--> Test passed')

    print('Testing init_guess_f')
    x0true = np.array([2, 3, 5, 5, 7, 9], dtype=float)
    x0 = init_guess_f(p0, trgt, v0, vf, psi0, psif, dt, deg)
    if not np.all(x0 == x0true):
        print('--> [!] Test failed')
    else:
        print('--> Test passed')

    print('Testing init_guess_m')
    x0true = np.array([], dtype=float)
    x0 = init_guess_m(p0, trgt, v0, vf, psi0, dt, deg)

    cpts1 = np.array([[0, 1, 2, 3, 4, 5],
                      [3, 4, 6, 2, 7, 9]], dtype=float)
    cpts2 = np.array([[5, 4, 3, 2, 1, 0],
                      [8, 3, 6, 6, 2, 5]], dtype=float)
