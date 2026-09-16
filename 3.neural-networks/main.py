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