from Datasets import MNIST
from Models import AlexNet

import numpy as np

def main():
    data = MNIST.get_data()
    model = AlexNet.Model(data)
    print(model)
    model.train()

if __name__ == "__main__":
    main()