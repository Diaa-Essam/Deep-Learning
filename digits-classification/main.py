import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits # Optical Recognition of Handwritten Digits dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


LEARNING_RATE = 0.1
EPOCHS = 1000
INPUT_SIZE = 64
OUTPUT_SIZE = 10

digits = load_digits()
x, y = digits.data, digits.target

# Dataset: 1797 samples, 64 features each (8x8 pixel images flattened)
# 10 classes (digits 0-9)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


y_one_hot = np.zeros((y.size, y.max() + 1))
y_one_hot[np.arange(y.size), y] = 1


def split_data(x, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, 
        test_size=test_size, 
        random_state=random_state
        )  
    
    return X_train, X_test, y_train, y_test   



x_train, x_test, y_train, y_test = split_data(
    x_scaled,
    y_one_hot
)

x_train_lr, x_test_lr, y_train_int, y_test_int = split_data(x_scaled, y)

class NeuralLayer:
    def __init__(self, input_size, output_size, learning_rate=LEARNING_RATE):
        """
        Initialize a single neural network layer.

        Parameters:
            input_size (int): number of input features to this layer
            output_size (int): number of output units (neurons) in this layer
            learning_rate (float): step size for gradient descent updates

        Sets:
            self.weights: ndarray of shape (input_size, output_size),
                           initialized with small random values
            self.bias: ndarray of shape (1, output_size), initialized to zeros
            self.learning_rate: stored for use in backward()
        """
        # your code here
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def sigmoid(self, x, deriv=False):
        result = 0
        if deriv == False:
            result =  1 / (1 + np.exp(-x))
        else: 
            s = self.sigmoid(x)
            result = s * (1 - s)

        return result

    def forward(self, X):
        """
        :type X: numpy.ndarray, shape (batch_size, input_size)
        :rtype: numpy.ndarray, shape (batch_size, output_size)
        """
        self.input = X # cache input for backward()

        
        Z = X @ self.weights + self.bias # Linear transformation: Z = XW + b

        # Apply sigmoid element-wise (works on whole matrix)
        # A = self.sigmoid(Z) (C backend), vertorized
        sigmoid_Z = []
        for row in Z:
            sig_row = [self.sigmoid(val) for val in row]
            sigmoid_Z.append(sig_row)

        A = np.array(sigmoid_Z)
        self.output = A

        return A

    def backward(self, y):
        """
        :type y: numpy.ndarray, shape (batch_size, output_size)
        :rtype: None  (updates self.weights and self.bias in place)
        """
        delta = (self.output - y) * self.output * (1 - self.output)

        grad_W = self.input.T @ delta

        grad_b = np.sum(delta, axis=0, keepdims=True)

        # Update weights and bias
        self.weights -= self.learning_rate * grad_W
        self.bias -= self.learning_rate * grad_b



# 3. Train Model Until Convergence
def train(layer, X_train, y_train, epochs=EPOCHS):
    """
    :type layer: NeuralLayer (already initialized)
    :type X_train: numpy.ndarray, shape (n_samples, input_size)
    :type y_train: numpy.ndarray, shape (n_samples, output_size)
    :type epochs: int
    :rtype: list of float — the loss recorded at each epoch
    """
    losses = []
    for epoch in range(EPOCHS + 1):
        A = layer.forward(X_train)
        loss = np.mean((A - y_train) ** 2)
        layer.backward(y_train)
        losses.append(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss:.4f}")

    return losses


# Create layer
layer = NeuralLayer(input_size=INPUT_SIZE , output_size=OUTPUT_SIZE , learning_rate=LEARNING_RATE )

# Train
losses = train(layer, x_train, y_train, epochs=EPOCHS)

# for epoch in range(len(losses)):
#     print(f"Epoch {epoch}: loss = {losses[epoch]:.4f}")


# 4.Predict and Score Accuracy
def evaluate(layer, X_test, y_test):
    """
    :type layer: NeuralLayer (already trained)
    :type X_test: numpy.ndarray, shape (n_samples, input_size)
    :type y_test: numpy.ndarray, shape (n_samples, output_size) — one-hot
    :rtype: float — accuracy as a fraction between 0 and 1
    """
    
    # A = layer.forward(X_test)
    # def argmax(row):
    #     maxIndex = 0
    #     maxVal = -1
    #     for i in range(len(row)):
    #         if row[i] > maxVal:
    #             maxVal = row[i]
    #             maxIndex = i

    #     return maxIndex
    
    # count = 0

    # for index in range(len(y_test)):
    #     predict_class = argmax(A[index])
    #     actual_class = argmax(y_test[index])

    #     if predict_class == actual_class:
    #         count += 1

    # return count / len(y_test)

    # Optimized Implementation
    A = layer.forward(X_test)
    return np.mean(np.argmax(A, axis=1) == np.argmax(y_test, axis=1))

# 5.Baseline Model Comparison
def train_and_evaluate_logistic(X_train, X_test, y_train_labels, y_test_labels):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train_labels)
    return model.score(X_test, y_test_labels)

nn_score = evaluate(layer, x_test, y_test)
lr_score = train_and_evaluate_logistic(x_train_lr, x_test_lr, y_train_int, y_test_int)

# Epoch 0: loss = 0.2506
# Epoch 100: loss = 0.0056
# Epoch 200: loss = 0.0038
# Epoch 300: loss = 0.0035
# Epoch 400: loss = 0.0034
# Epoch 500: loss = 0.0033
# Epoch 600: loss = 0.0031
# Epoch 700: loss = 0.0030
# Epoch 800: loss = 0.0030
# Epoch 900: loss = 0.0028
# Epoch 1000: loss = 0.0028
# Nerual Network: 0.9527777777777777
# Logistic Regression: 0.9722222222222222

print(f'Nerual Network: {nn_score}')
print(f'Logistic Regression: {lr_score}')

def lr_sweep(X_train, y_train, X_test, y_test, learning_rates, epochs=EPOCHS):
    """
    :type learning_rates: list of float, e.g. [0.001, 0.01, 0.1, 1.0]
    :rtype: dict mapping each learning_rate -> final test accuracy
    """
    result = {}
    for lr in learning_rates:
        layer = NeuralLayer(INPUT_SIZE, OUTPUT_SIZE, learning_rate=lr)
        train(layer,X_train, y_train, EPOCHS)
        acc = evaluate(layer, X_test, y_test)
        result[lr] = acc

    return result

results = lr_sweep(
    x_train, y_train, x_test, y_test,
    learning_rates=[0.001, 0.01, 0.1, 1.0, 5.0],
    epochs=1000
)
print("\n=== Learning Rate Sweep ===")
for lr, acc in results.items():
    print(f"η = {lr:<6} → accuracy = {acc:.4f}")




    
