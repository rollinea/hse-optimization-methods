import numpy as np
from matplotlib import pyplot as plt

import optimization
import oracles

n, m = 500, 1000
A = np.random.uniform(low=-100, high=100, size=(n, m))
b = np.random.uniform(low=-100, high=100, size=n)

regcoef = 1
oracle = oracles.create_lasso_prox_oracle(A, b, regcoef)
x_0 = np.zeros(m)

def plot(history, name):
    name_ = name
    name = ' '.join(name.split('_')).title()
    y = history['iterations']
    x = np.array(list(range(len(y))))
    plt.figure()
    plt.title(name, fontsize=14)
    plt.plot(x, y)
    plt.xlabel('Method Iterations', fontsize=14)
    plt.ylabel('Line Search Iterations', fontsize=14)
    plt.grid()
    plt.savefig('experiment_2/' + name_ + '.png')

for method in [optimization.proximal_gradient_method,
               optimization.proximal_fast_gradient_method]:
    print(f'Starting {method.__name__}!')
    x_star, message, history = method(oracle, x_0, trace=True)
    print(f'Result is: {message}')
    print(f'Average number of iterations of the linear search: {np.mean(history["iterations"][1:])}')
    print(f'Minimal duality gap: {round(min(history["duality_gap"]), 4)}')
    plot(history, method.__name__)
    print('\n')