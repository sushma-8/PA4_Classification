import numpy as np

MAX_ITERATIONS = 70


class Perceptron:

    def __init__(self):
        self.weights = {}
        self.bias = 0
        self.count = 0
        self.input_data = None

    def parse_input(self):
        data_points = np.loadtxt(open('classification.txt', 'r'), delimiter='\t', dtype='str')
        self.input_data = np.array([x.split(',')[:-1] for x in data_points], dtype=np.float)
        self.count = len(self.input_data)

    def shuffle_vectors(self):
        index_shuffle = np.random.permutation(self.input_data.shape[0])
        self.input_data = self.input_data[index_shuffle]

    def learn_weights(self):
        self.weights = np.zeros(3, )

        for _ in range(MAX_ITERATIONS):
            self.shuffle_vectors()
            for row in self.input_data:
                activation = np.dot(row[:-1], self.weights)
                activation = np.add(activation, self.bias)

                if (row[-1] * activation) <= 0:
                    self.weights = np.add(self.weights, np.dot(row[:-1], row[-1]))
                    self.bias = np.add(self.bias, row[-1])
        print(f'Weights:{self.weights}\nBias:{self.bias}')

    def test_model(self):
        correct_predictions = 0
        for row in self.input_data:
            value = np.dot(row[:-1], self.weights)
            value += self.bias
            if (value > 0 and row[-1] == 1) or (value < 0 and row[-1] == -1):
                correct_predictions += 1
        print(f'Correct Predictions:{correct_predictions}')


if __name__ == "__main__":
    model = Perceptron()
    model.parse_input()
    model.learn_weights()
    model.test_model()
