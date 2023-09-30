from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from time import time
from datetime import datetime
from oracles import *


def update_history(x_k, f_k, dt, duality_gap, history, iterations=0):
    history['func'].append(f_k)
    history['time'].append((datetime.now() - dt).total_seconds())
    history['duality_gap'].append(duality_gap)
    history['iterations'].append(iterations)
    if x_k.size <= 2:
        history['x'].append(x_k)
    return history


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.

    history = defaultdict(list) if trace else None

    if max_iter is None:
        max_iter = max_iter

    stop = tolerance

    x_k, a_k = np.copy(x_0), np.copy(alpha_0)
    x_best, f_best = None, None

    dt = datetime.now()

    for step in range(max_iter + 1):
        f_k = oracle.func(x_k)
        duality_gap = oracle.duality_gap(x_k)
        if f_best is None or f_best > f_k:
            x_best, f_best = np.copy(x_k), np.copy(f_k)

        if trace:
            update_history(x_k, f_k, dt, duality_gap, history)

        if duality_gap <= stop:
            return x_k, 'success', history

        if step == max_iter:
            break

        a_k = alpha_0 /  float(np.sqrt(step + 1))
        subgrad_k = oracle.subgrad(x_k)
        x_k -= a_k * subgrad_k / np.linalg.norm(subgrad_k)

    return x_best, 'iterations exceeded', history


def proximal_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                             max_iter=1000, trace=False, display=False):
    """
    Gradient method for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of objective function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.

    history = defaultdict(list) if trace else None
    x_k, L_k = np.copy(x_0), np.copy(L_0)
    iterations = 0
    dt = datetime.now()

    for step in range(max_iter + 1):
        f_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)
        duality_gap_k = oracle.duality_gap(x_k)

        if trace:
            update_history(x_k, f_k, dt, duality_gap_k, history, iterations)

        if duality_gap_k <= tolerance:
            return x_k, 'success', history

        if step == max_iter:
            break

        iterations = 0
        while True:
            iterations += 1
            x_new = oracle.prox(x_k - grad_k / L_k, 1 / L_k)
            if oracle._f.func(x_new) > oracle._f.func(x_k) +\
                    grad_k @ (x_new - x_k) + (L_k / 2.) * np.linalg.norm(x_new - x_k) ** 2:
                L_k *= 2.
            else:
                L_k /= 2.
                x_k = np.copy(x_new)
                break

    return x_k, 'iterations exceeded', history


def proximal_fast_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                                  max_iter=1000, trace=False, display=False):
    """
    Fast gradient method for composite minimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of objective function values phi(best_point) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
    """
    # TODO: Implement
    history = defaultdict(list) if trace else None
    x_k, L_k = np.copy(x_0), np.copy(L_0)
    x_best, f_best = np.copy(x_0), None
    A_k, v_k, y_k = 0, np.copy(x_k), np.copy(x_k)
    sum_a_grad = 0
    iterations = 0
    dt = datetime.now()

    for step in range(max_iter + 1):
        f_k = oracle.func(x_k)
        duality_gap = oracle.duality_gap(x_k)

        if f_best is None or f_k < f_best:
            f_best = np.copy(f_k)
            x_best = np.copy(x_k)

        if trace:
            update_history(x_k, f_k, dt, duality_gap, history, iterations)

        if duality_gap <= tolerance:
            return x_best, 'success', history

        if step == max_iter:
            break

        iterations = 0
        while True:
            iterations += 1
            a_k = (1. + np.sqrt(1. + 4. * L_k * A_k)) / (2. * L_k)
            A_new = A_k + a_k
            y_k = (A_k * x_k + a_k * v_k) / A_new
            y_grad_k = oracle.grad(y_k)
            sum_a_grad_new = sum_a_grad + a_k * y_grad_k
            v_new = oracle.prox(x_0 - sum_a_grad_new, A_new)
            x_new = (A_k * x_k + a_k * v_new) / A_new

            if oracle._f.func(x_new) > oracle._f.func(y_k) \
                    + y_grad_k @ (x_new - y_k) + L_k * np.linalg.norm(x_new - y_k) ** 2 / 2.:
                L_k *= 2.
            else:
                x_k = np.copy(x_new)
                A_k = np.copy(A_new)
                v_k = np.copy(v_new)
                sum_a_grad = np.copy(sum_a_grad_new)
                L_k /= 2.
                break

    return x_best, 'iterations exceeded', history
