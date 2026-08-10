def perceptron(x, y, w1, w2, threshold):
    total = w1 * x + w2 * y

    if total >= threshold:
        return 1
    return 0


print(perceptron(0, 0, 1, 1, 2))  # 0
print(perceptron(0, 1, 1, 1, 2))  # 0
print(perceptron(1, 0, 1, 1, 2))  # 0
print(perceptron(1, 1, 1, 1, 2))  # 1

print('==============================')

print(perceptron(0, 0, 1, -1, 0))
print(perceptron(0, 1, 1, -1, 0))
print(perceptron(1, 0, 1, -1, 0))
print(perceptron(1, 1, 1, -1, 0))