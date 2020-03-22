import numpy as np
import math
import matplotlib.pyplot as plt

MAX_ITERATIONS = 7000
learning_rate  = 0.01

class Logistic_Regression:

    def __init__(self):
        self.weights = {}
        self.bias = 0
        self.count = 0
        self.input_data = None

    def parse_input(self):
        data_points = np.loadtxt(open('classification.txt', 'r'), delimiter='\t', dtype='str')
        self.input_data = np.array([x.split(',') for x in data_points], dtype=np.float)
        self.input_data = np.delete(self.input_data, 3, 1)
        self.X = self.input_data
        self.Y = self.input_data[:,3]
        self.input_data = self.input_data[:,0:3]
        self.shape = np.shape(self.input_data)
        self.input_data = np.concatenate((np.ones((self.shape[0],1)), self.input_data), axis = 1)

    def shuffle_vectors(self):
        index_shuffle = np.random.permutation(self.input_data.shape[0])
        self.input_data = self.input_data[index_shuffle]

    def get_random_vector(self):

        return self.input_data[np.random.randint(self.count),:].reshape((4,))


    def activation_function(self, s):

        return 1 / (1 + np.exp(-s))


    def learn_weights(self):
        self.weights = np.random.rand(1, self.shape[1]+1)
        count = 0
        for _ in range(MAX_ITERATIONS):

            count += 1
            print(f"Iteration : {count}")
            s = self.activation_function(np.dot(self.weights, self.input_data.T))#, self.weights.T))

            dw = (1/self.input_data.shape[0])*(np.dot(self.input_data.T, (s-self.Y.T).T))
            self.weights = self.weights - (learning_rate * (dw.T))

        print(f'Weights:{self.weights}\nBias:{self.bias}')

    def test_model(self):
        correct_predictions = 0
        for row in self.X:
            value = np.dot(row, self.weights.T)
            if (value > 0.5 and row[-1] == 1) or (value < 0.5 and row[-1] == -1):
                correct_predictions += 1
        print(f'Correct Predictions:{correct_predictions/self.input_data.shape[0]}')

        return correct_predictions


    def plot(self):

        x = np.arange(0, len(self.mis_classification), 1)
        plt.plot(x, self.mis_classification)
        plt.xlabel('# of Iterations')
        plt.ylabel('# of miss classifications')
        plt.show()

if __name__ == "__main__":
    model = Logistic_Regression()
    model.parse_input()
    model.learn_weights()
    model.test_model()
    #model.plot()
