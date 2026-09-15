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

    s = sigmoid()
    return s - (1 - s)