import numpy as np

# 6.Sigmoid Activation and Derivative
def sigmoid(x):
    """
    :type x: float or numpy.ndarray
    :rtype: same type as x
    """

    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    """
    :type x: float or numpy.ndarray
    :rtype: same type as x
    """

    s = sigmoid(x)
    return s * (1 - s)

# 9. Mean Squared Error Loss
def mse_loss(y_true, y_pred):
    """
    :type y_true: numpy.ndarray, shape (n,)
    :type y_pred: numpy.ndarray, shape (n,)
    :rtype: float — mean squared error across all samples
    """
    # mse = ((y_true - y_pred) ** 2).mean()

    mse = 0
    n = len(y_true)
    for i in range(n):
        mse += (y_true[i] - y_pred[i]) ** 2
    mse /= n

    return mse

# 7. Single Neuron Feedforward
class Neuron:
    def __init__(self, weights, bias):
        """
        :type weights: numpy.ndarray, shape (n_inputs,)
        :type bias: float
        """
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        """
        :type inputs: numpy.ndarray, shape (n_inputs,)
        :rtype: float — the neuron's output after sigmoid activation
        """

        # z = np.dot(self.weights, inputs) + self.bias

        z = 0
        for i in range(len(self.weights)):
            z += (inputs[i] * self.weights[i])
        z += self.bias
        
        return sigmoid(z)

class OurNeuralNetwork:
    def __init__(self):
        """
        All three neurons share the same weights w = [0, 1] and bias b = 0.
        Creates: self.h1, self.h2, self.o1 (all Neuron instances)
        """
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        """
        :type x: numpy.ndarray, shape (2,)
        :rtype: tuple (o1, sum_h1, h1, sum_h2, h2, sum_o1)
        """
        # Hidden neuron h1
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        # Hidden neuron h2
        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        # Output neuron o1
        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)

        return o1, sum_h1, h1, sum_h2, h2, sum_o1



net = OurNeuralNetwork()
o1, sh1, h1, sh2, h2, so1 = net.feedforward(np.array([-2, -1]))
print(f"o1 = {o1:.4f}")