"""
Perceptron

Contributors:

Niroop Ramdas Sagar
USC ID: 4897621292
ramdassa@usc.edu

Sushma Mahadevaswamy
USC ID: 3939734806
mahadeva@usc.edu

"""
import numpy as np

MAX_ITERATIONS = 7000


class Perceptron:

    def __init__(self):
        self.weights = {}
        self.bias = 0
        self.count = 0
        self.input_data = None
        self.rand_scratch_card = []

    def parse_input(self):
        data_points = np.loadtxt(open('classification.txt', 'r'), delimiter='\t', dtype='str')
        self.input_data = np.array([x.split(',')[:-1] for x in data_points], dtype=np.float)
        self.count = len(self.input_data)

    def shuffle_vectors(self):
        index_shuffle = np.random.permutation(self.input_data.shape[0])
        self.input_data = self.input_data[index_shuffle]

    def get_random_vector(self):
        idx = np.random.randint(self.count)
        while idx in self.rand_scratch_card:
            idx = np.random.randint(self.count)
        self.rand_scratch_card.append(idx)

        return self.input_data[idx,:].reshape((4,))

    def learn_weights(self):
        self.weights = np.zeros(3, )

        is_converged = False
        for _ in range(MAX_ITERATIONS):
            self.rand_scratch_card = []
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

            if is_converged:
                print(f"Converged")
                return


    def test_model(self):
        correct_predictions = 0
        for row in self.input_data:
            value = np.dot(row[:-1], self.weights)
            value += self.bias
            if (value > 0 and row[-1] == 1) or (value < 0 and row[-1] == -1):
                correct_predictions += 1

        return correct_predictions

if __name__ == "__main__":
    model = Perceptron()
    model.parse_input()
    model.learn_weights()
    print(f'Weights:{model.weights}\nBias:{model.bias}')
    print(f'Accuracy : {(model.test_model()/model.count) * 100} %')
