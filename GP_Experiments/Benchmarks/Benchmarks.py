"""Benchmarks functions from: https://dev.heuristiclab.com/trac.fcgi/blog/gkronber/symbolic_regression_benchmark"""

import numpy as np


def Benchmarks(bench):
    if bench == "keijzer 1":
        f = lambda x: 0.3 * x * np.sin(2*np.pi*x)
        interval = np.linspace(-1, 1, 21)
        terminals = 1
    if bench == "keijzer 2":
        f = lambda x: 0.3 * x * np.sin(2*np.pi*x)
        interval = np.linspace(-2, 2, 21)
        terminals = 1
    if bench == "keijzer 3":
        f = lambda x: 0.3 * x * np.sin(2*np.pi*x)
        interval = np.linspace(-3, 3, 21)
        terminals = 1
    elif bench == "keijzer 4":
        f = lambda x: x**3 + np.exp(-x) * np.cos(x) * np.sin(x) * (np.sin(x)**2 * np.cos(x) - 1)
        interval = np.linspace(0, 10, 11)
        terminals = 1
    elif bench == "keijzer 5":
        f = lambda x, y, z: (30 * x * z)/((x - 10) * y**2)
        interval1 = np.linspace(-1, 1, 1000)
        interval2 = interval1
        interval3 = np.linspace(1, 2, 1000)
        interval = np.vstack((interval1, interval2, interval3))
        terminals = 3
    elif bench == "keijzer 6":
        f = lambda x: [sum(i) for i in range(x)]
        interval = np.linspace(1, 50, 50)
        terminals = 1
    elif bench == "keijzer 7":
        f = lambda x: np.log(x)
        interval = np.linspace(1, 100, 100)
        terminals = 1
    elif bench == "keijzer 8":
        f = lambda x: np.sqrt(x)
        interval = np.linspace(0, 100, 101)
        terminals = 1
    elif bench == "keijzer 10":
        f = lambda x, y: x**y
        interval1 = np.linspace(0, 1, 100)
        interval2 = interval1
        interval = np.vstack((interval1, interval2))
        terminals = 2
    elif bench == "keijzer 11":
        f = lambda x, y: x * y + np.sin((x - 1) * (y - 1))
        interval1 = np.linspace(-3, 3, 20)
        interval2 = interval1
        interval = np.vstack((interval1, interval2))
        terminals = 2
    elif bench == "keijzer 13":
        f = lambda x, y: 6 * np.sin(x) * np.cos(y)
        interval1 = np.linspace(-3, 3, 20)
        interval2 = interval1
        interval = np.vstack((interval1, interval2))
        terminals = 2
    elif bench == "keijzer 14":
        f = lambda x, y: 8 / (2 + x**2 + y**2)
        interval1 = np.linspace(-3, 3, 20)
        interval2 = np.linspace(-3, 3, 20)
        interval = np.vstack((interval1, interval2))
        terminals = 2
    elif bench == "keijzer 15":
        f = lambda x, y: x**3/5 + y**3/2 - y - x
        interval1 = np.linspace(-3, 3, 20)
        interval2 = np.linspace(-3, 3, 20)
        interval = np.vstack((interval1, interval2))
        terminals = 2
    elif bench == "korns 1":
        f = lambda x: 1.57 + 24.3 * x
        interval = np.linspace(-50, 50, 10000)
        terminals = 1
    elif bench == "korns 2":
        f = lambda x, y, z: 0.23 + 14.2 * ((x + y) / 3 * z)
        interval1 = np.linspace(-50, 50, 10000)
        interval2 = interval1
        interval3 = interval1
        interval = np.vstack((interval1, interval2, interval3))
        terminals = 3
    elif bench == "korns 3":
        f = lambda x, y, z, w: -5.41 + 4.9 * ((z - x + y/w) / (3 * w))
        interval1 = np.linspace(-50, 50, 10000)
        interval2 = interval1
        interval3 = interval1
        interval4 = interval1
        interval = np.vstack((interval1, interval2, interval3, interval4))
        terminals = 4
    elif bench == "korns 4":
        f = lambda x: -2.3 + 0.13 * np.sin(x)
        interval1 = np.linspace(-50, 50, 10000)
        interval2 = interval1
        interval3 = interval1
        interval = np.vstack((interval1, interval2, interval3))
        terminals = 4
    return f, interval, terminals