import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculateParameters(xData, yData):
    matrix = np.array([[len(xData), np.sum(xData)],
                             [ np.sum(xData), np.sum(xData * xData)]])
    vector = np.array([[np.sum(yData)],
                       [np.sum(xData * yData)]])
    return np.linalg.solve(matrix, vector)

x = np.linspace(0, 10, 20)
err = np.random.uniform(-0.25, 0.25, len(x))
y = x + err
params = calculateParameters(x, y)
fit = params[0] + params[1] * x

plt.scatter(x, y)
plt.plot(x, fit, color='red')
plt.show()
