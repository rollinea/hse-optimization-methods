import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt
from sklearn.datasets import load_svmlight_file

import optimization
import oracles

seed = 42
np.random.seed(seed=seed)

n, m = 500, 1000
A = np.random.uniform(low=-100, high=100, size=(n, m))
b = np.random.uniform(low=-100, high=100, size=n)

regcoef = 1
oracle = oracles.create_lasso_nonsmooth_oracle(A, b, regcoef)
start_points = [np.random.uniform(-1, 1, m), np.random.uniform(-5, 5, m), np.random.uniform(-100, 100, m)]
alphas = [0.1, 0.5, 1.0, 1.5, 2]
labels = ['a_0 = 0.1', 'a_0 = 0.5', 'a_0 = 1.0', 'a_0 = 1.5', 'a_0 = 2']


def plot(i, histories, labels):
    name_ = f'subgradient_method_{i}'
    plt.figure()
    plt.xlabel('Method Iterations', fontsize=14)
    plt.ylabel('Duality Gap', fontsize=14)
    plt.grid()
    for history, label in zip(histories, labels):
        y = history['duality_gap']
        x = np.arange(len(history['duality_gap']))
        plt.plot(x, y, label=label)

    plt.legend(loc='best')
    plt.savefig('experiment_1/' + name_ + '.png')

for i, x_0 in enumerate(start_points):
    print(f'Start point: {i}')
    histories = []
    for alpha in alphas:
        print(f'Alpha: {alpha}')
        x_star, message, history = optimization.subgradient_method(oracle, x_0, alpha_0=alpha, trace=True)
        histories.append(history)
        print(f'Result: {message}, minimal duality_gap: {round(min(history["duality_gap"]), 4)}')
    print('\n')
    plot(i, histories, labels)
