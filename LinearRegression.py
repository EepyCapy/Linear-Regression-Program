import numpy as np

class LinearRegression():
    def __init__(self, xData, yData):
        self.xData = xData
        self.yData = yData
        self.parameters = np.array([])

    def multidLS(self):
        print(self.xData)
        X = np.insert(self.xData, 0, np.ones(len(self.yData)), axis=1)
        matrix = X.T @ X
        vector = X.T @ self.yData
        self.parameters = np.linalg.solve(matrix, vector)

    def predictedValues(self, axis):
        yPred = self.parameters[0] * np.ones(len(self.yData))
        yPred = yPred + self.parameters[axis + 1] * self.xData[:, axis]
        print(yPred)
        return yPred
