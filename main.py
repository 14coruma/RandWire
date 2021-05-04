from Datasets import MNIST
from Models import AlexNet, TinyCNN, RandWire

from tensorflow.keras.utils import plot_model

import numpy as np

def main():
    data = MNIST.get_data(n=8000, m=800)
    #model = AlexNet.Model(data)
    #model = TinyCNN.Model(data)
    model = RandWire.Model(data)
    model.train()
    model.test()
    #model.summary()
    #plot_model(model.model, to_file='TinyCNN.png')


if __name__ == "__main__":
    main()