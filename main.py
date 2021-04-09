from Datasets import MNIST
from Models import AlexNet, TinyCNN

import numpy as np

def main():
    data = MNIST.get_data(n=4000, m=400)
    #model = AlexNet.Model(data)
    model = TinyCNN.Model(data)
    model.train()
    model.summary()
    model.test()

if __name__ == "__main__":
    main()