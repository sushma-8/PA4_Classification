import numpy as np
import math
import matplotlib.pyplot as plt

class Linear_regression:

    def __init__(self):
        self.weights = {}
        self.bias = 0
        self.count = 0
        self.input_data = None

    def parse_input(self):
        data_points = np.loadtxt(open('linear-regression.txt', 'r'), delimiter='\t', dtype='str')
        self.input_data = np.array([x.split(',') for x in data_points], dtype=np.float)
        self.XY = np.mat(self.input_data[:,0:2])
        self.Z  = self.input_data[:,2]
        self.Z  = np.mat(self.Z[:, np.newaxis])
        self.count = len(self.input_data)
        self.shape = np.shape(self.XY)
        self.XY = np.concatenate((np.ones((self.shape[0],1)), self.XY), axis = 1)

    def learn_weights(self):
        self.weights = np.zeros(3, )
        xt_x = self.XY.T * self.XY
        if (np.linalg.det(xt_x) == 0):
            print("Singular Matrix. Cannot find Inverse")
        self.weights = xt_x.I * (self.XY.T * self.Z)
        self.weights  = self.weights.T

        print(f'Weights:{self.weights}')


if __name__ == "__main__":
    model = Linear_regression()
    model.parse_input()
    model.learn_weights()
