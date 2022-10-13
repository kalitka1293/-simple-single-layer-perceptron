import numpy as np
class Perceptron:
    def __init__(self, x, y, threshold = 0.5, learning_rate = 0.1, max_epochs = 10):
        self.x = x
        self.y = y
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def initialize(self, init_type = 'zeros'):
        if init_type == 'random':
            self.weights = np.random.rand(len(self.x[0]))*0.05

        if init_type == 'zeros':
            self.weights = np.zeros(len(self.x[0]))

    def train(self):
        epoch = 0
        while True:
            error_count = 0
            epoch += 1
            for (x, y) in zip(self.x, self.y):
                error_count = self.train_observation(x, y, error_count)

            if error_count == 0:
                print('training successful')
                break

            if epoch > self.max_epochs:
                print('reached maximum epoch, no perfect prediction')
                break

    def train_observation(self, x, y, error_count):
        result = np.dot(x, self.weights) > self.threshold
        error = y - result
        print(error)
        if error != 0:
            error_count += 1
            for index, value in enumerate(x):
                self.weights[index] += self.learning_rate * error * value

                print(self.weights[index], value, index)

        return error_count
    def predict(self, x):
        return int(np.dot(x, self.weights) > self.threshold)

# x = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 1), (1, 0, 1), (1, 0, 1)]
# y = [1, 1, 0, 0, 1, 1]

x = [(1, 1, 1, 1), (0, 1, 1, 1), (1, 1, 0, 1), (1, 0, 1, 1), (1, 1, 1, 1), (0, 1, 1, 1), (0, 1, 1, 1), (1, 0, 0, 1), (1, 1, 1, 0)]
y = [1, 1, 1, 1, 1, 1, 1, 1, 0]


p = Perceptron(x, y)
p.initialize()
p.train()
print(p.predict((1,1,1, 1)))
print(p.predict((1,0,1, 0)))
print(p.weights)
zxc = (1, 1, 1, 0)
print(np.zeros(len(x[0])), 'test')


