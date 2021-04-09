from Datasets import MNIST
from Models import AlexNet, TinyCNN

import numpy as np

def main():
    data = MNIST.get_data()
    #model = AlexNet.Model(data)
    model = TinyCNN.Model(data)
    model.train()
    model.summary()
    model.test()

if __name__ == "__main__":
    main()