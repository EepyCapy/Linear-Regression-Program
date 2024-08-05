import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression():
    def __init__(self): #, xData, yData):
        self.xData = np.array([])
        self.yData = np.array([])
        self.parameters = np.array([])
        self.fig = plt.figure()
        if self.xData.ndim == 1:
            self.ax = self.fig.add_axes()
        elif self.xData.ndim == 2:
            self.ax = self.fig.add_subplot(projection='3d')

    def multidLS(self):
        if self.xData.ndim == 1:
            self.xData = np.reshape(self.xData, (len(self.xData), 1))
        X = np.insert(self.xData, 0, np.ones(len(self.yData)), axis=1)
        matrix = X.T @ X
        vector = X.T @ np.reshape(self.yData, (len(self.yData), 1))
        self.parameters = np.linalg.solve(matrix, vector)

    def plotData(self):
        if self.xData.ndim == 1:
            self.ax.scatter(self.xData, self.yData)
        elif self.xData.ndim == 2:
            self.ax.scatter(self.xData[:, 0], self.xData[:, 1], self.yData)
        else:
            print('Cannot plot more than 3 dimensions')

    def plotPrediction(self):
        if self.xData.ndim == 1:
            yPred = self.parameters[0] + self.parameters[1] * self.xData
            self.ax.plot(self.xData, yPred)
        elif self.xData.ndim == 2:
            x1, x2 = np.meshgrid(self.xData[:, 0], self.xData[:, 1])
            yPred = self.parameters[0] + self.parameters[1] * x1 + self.parameters[2] * x2
            self.ax.plot_surface(x1, x2, yPred, color='purple')
        else:
            print('Cannot plot more than 3 dimensions')

# Formulas for Simple Linear Regression
#def calculateParameters(xData, yData):
#    matrix = np.array([[len(xData)    , np.sum(xData)],
#                       [ np.sum(xData), np.sum(xData * xData)]])
#    vector = np.array([[np.sum(yData)],
#                       [np.sum(xData * yData)]])
#    return np.linalg.solve(matrix, vector)
