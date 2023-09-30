import numpy as np
from matplotlib import pyplot as plt

import optimization
import oracles
import itertools

np.random.seed(42)

n_ranges = [500, 1000]
m_ranges = [1000, 5000]
regcoefs = [0.1, 1, 5]
methods = [optimization.subgradient_method,
           optimization.proximal_gradient_method,
           optimization.proximal_fast_gradient_method]

def plot(histories, n, m, regcoef):
    name = f'n={n}, m={m}, regcoef={regcoef}'
    for x_name in ['Iterations', 'Time']:
        plt.figure()
        plt.title(name)
        for history, method in zip(histories, methods):
            method_name = ' '.join(method.__name__.split('_')).title()
            y = history['duality_gap']
            x = history['time'] if x_name == 'Time' else list(range(len(y)))
            plt.plot(x, y, label=method_name)
        plt.xlabel(x_name, fontsize=14)
        plt.ylabel('Duality gap', fontsize=14)
        plt.yscale('log')
        plt.legend(fontsize=10, loc='best')
        plt.grid()
        plt.savefig('experiment_3/' + name + '-' + x_name + '.png')
    plt.close('all')


for _, (n, m, regcoef) in enumerate(itertools.product(n_ranges, m_ranges, regcoefs)):
    print(f'Combinations: {n}, {m}, {regcoef}')
    print('\t\t')
    A = np.random.uniform(low=-100, high=100, size=(m, n))
    b = np.random.uniform(low=-100, high=100, size=m)
    x_0 = np.zeros(n)
    histories = []
    for i, method in enumerate(methods):
        print(method.__name__)
        if i == 0:
            oracle = oracles.create_lasso_nonsmooth_oracle(A, b, regcoef)
            x_star, message, history = method(oracle, x_0, trace=True)
        else:
            oracle = oracles.create_lasso_prox_oracle(A, b, regcoef)
            x_star, message, history = method(oracle, x_0, trace=True)
        print(f'Result: {message}, Minimal duality gap: {round(min(history["duality_gap"]), 4)}\n')
        histories.append(history)

    plot(histories, n, m, regcoef)
