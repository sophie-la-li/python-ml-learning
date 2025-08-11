import sys
import os
import random
import time

sys.path.append(os.path.join(sys.path[0], '..'))
import src.NeuralNetwork as NN

addition = NN.NeuralNetwork()
addition.inputSize = 2
addition.hiddenSize = 2
addition.hiddenNumber = 0
addition.outputSize = 1

for i in range(10000):
    a: int = random.randint(0,10)
    b: int = random.randint(0,10)
    e: int = a+b
    addition.train([a,b], [a+b])

while True:
    for i in range(1000):
        a: int = random.randint(0,10)
        b: int = random.randint(0,10)
        e: int = a+b
        r: float = addition.train([a,b], [a+b])[0]
        print(a, "+", b, "=", r, abs(r - e), abs(r - e) < 0.1)
        time.sleep(0.5)

