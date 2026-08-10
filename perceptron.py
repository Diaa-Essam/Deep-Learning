def perceptron(inputs, weights, threshold):
    total = 0

    for x, w in zip(inputs, weights):
        total += x * w

    if total >= threshold:
        return 1
    return 0

X = 1
Y = 0
Z = 1
A = 0

p1 = perceptron([X, Y], [1, 1], 2)
p2 = perceptron([X, Z], [1, 1], 2)
p3 = perceptron([Y, A], [-1, 1], 1)
p4 = perceptron([X, Z, A], [1, 1, 1], 2)

p5 = perceptron([p1, p2], [1, -1], 0)
p6 = perceptron([p3, p4], [1, 1], 1)

p7 = perceptron([p5, p6], [1, 1], 2)


print(p1, p2, p3, p4, p5, p6, p7)
print(p7)