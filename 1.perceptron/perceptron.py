def perceptron(inputs, weights, threshold):
    total = 0

    for x, w in zip(inputs, weights):
        total += x * w

    if total >= threshold:
        return 1
    return 0


def circuit(X,Y,Z,A):
    p1 = perceptron([X, Y], [1, 1], 2)
    p2 = perceptron([X, Z], [1, 1], 2)
    p3 = perceptron([Y, A], [-1, 1], 1)
    p4 = perceptron([X, Z, A], [-1, 1, 1], 2)

    p5 = perceptron([p1, p2], [1, -1], 0)
    p6 = perceptron([p3, p4], [1, 1], 1)

    p7 = perceptron([p5, p6], [1, 1], 2)

    return p7


print(" ------------------ Complete Truth Table ------------------")
for X in [0,1]:
    for Y in [0,1]:
        for Z in [0,1]:
            for A in [0,1]:
                print(f"X={X}, Y={Y}, Z={Z}, A={A} -> Output={circuit(X,Y,Z,A)}")

