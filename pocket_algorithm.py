import numpy as np
import math
import matplotlib.pyplot as plt

MAX_ITERATIONS = 7000


class Pocket:

    def __init__(self):
        self.weights = {}
        self.bias = 0
        self.count = 0
        self.input_data = None
        self.best_weights = {}
        self.mis_classification = []

    def parse_input(self):
        data_points = np.loadtxt(open('classification.txt', 'r'), delimiter='\t', dtype='str')
        self.input_data = np.array([x.split(',') for x in data_points], dtype=np.float)
        self.input_data = np.delete(self.input_data, 3, 1)
        self.count = len(self.input_data)

    def shuffle_vectors(self):
        index_shuffle = np.random.permutation(self.input_data.shape[0])
        self.input_data = self.input_data[index_shuffle]

    def get_random_vector(self):

        return self.input_data[np.random.randint(self.count),:].reshape((4,))

    def learn_weights(self):
        self.weights = np.zeros(3, )
        self.best_weights = np.zeros(3, )

        is_converged = False
        max_predictions = -math.inf

        count = 0
        for _ in range(MAX_ITERATIONS):
            #self.shuffle_vectors()
            constraint_violated = False
            while (constraint_violated != True):
                row = self.get_random_vector()
                activation = np.dot(row[:-1], self.weights)
                activation = np.add(activation, self.bias)

                if (row[-1] * activation) <= 0:
                    self.weights = np.add(self.weights, np.dot(row[:-1], row[-1]))
                    self.bias = np.add(self.bias, row[-1])
                    constraint_violated = True

            cur_predictions = self.test_model()

            if cur_predictions == self.count:
                is_converged = True

            if cur_predictions >= max_predictions:
                max_predictions   = cur_predictions
                self.best_weights = self.weights
            self.mis_classification.append(self.count - cur_predictions)

            if is_converged:
                print(f"Converged")
                return
                
        print(f'Weights:{self.weights}\nBias:{self.bias}')
        print(f'Max predictions : {max_predictions}')

    def test_model(self):
        correct_predictions = 0
        for row in self.input_data:
            value = np.dot(row[:-1], self.weights)
            value += self.bias
            if (value > 0 and row[-1] == 1) or (value < 0 and row[-1] == -1):
                correct_predictions += 1
        print(f'Correct Predictions:{correct_predictions}')

        return correct_predictions


    def plot(self):

        x = np.arange(0, len(self.mis_classification), 1)
        plt.plot(x, self.mis_classification)
        plt.xlabel('# of Iterations')
        plt.ylabel('# of miss classifications')
        plt.show()

if __name__ == "__main__":
    model = Pocket()
    model.parse_input()
    model.learn_weights()
    model.test_model()
    model.plot()
