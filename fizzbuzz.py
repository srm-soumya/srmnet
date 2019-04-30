"""
Number n,
n % 15 == 0, fizzbuzz
n % 5  == 0, buzz
n % 3  == 0, fizz
"""

from typing import List
import numpy as np

from srmnet.layers import Linear, Tanh
from srmnet.nn import NeuralNet
from srmnet.train import train
from srmnet.optim import SGD


def fizzbuzz_encode(n: int) -> List[int]:
    if n % 15 == 0:
        return np.array([0, 0, 0, 1])
    elif n % 5 == 0:
        return np.array([0, 0, 1, 0])
    elif n % 3 == 0:
        return np.array([0, 1, 0, 0])
    else:
        return np.array([1, 0, 0, 0])

def binary_encode(x: int) -> List[int]:
    return [x >> i & 1 for i in range(10)]

inputs = np.array([binary_encode(x) for x in range(101, 1024)])
targets = np.array([fizzbuzz_encode(x) for x in range(101, 1024)])

net = NeuralNet([
    Linear(input_size=10, output_size=20),
    Tanh(),
    Linear(input_size=20, output_size=4)
])


train(net,
      inputs,
      targets,
      num_epochs=1,
      optimizer=SGD(lr=2e-3))

for x in range(1, 101):
    predicted = net.forward(binary_encode(x))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizzbuzz_encode(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_idx], labels[actual_idx])