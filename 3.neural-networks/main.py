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
        weights = [0, 1]
        self.h1 = Neuron(weights, 0)
        self.h2 = Neuron(weights, 0)
        self.o1 = Neuron(weights, 0)

    def feedforward(self, x):
        """
        :type x: numpy.ndarray, shape (2,)
        :rtype: float — the network's final output
        """
        
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        nx = np.array([out_h1, out_h2])

        return self.o1.feedforward(nx)