from Datasets import MNIST
from Models import AlexNet, TinyCNN

from tensorflow.keras.utils import plot_model

import numpy as np

def main():
    data = MNIST.get_data(n=4000, m=400)
    model = AlexNet.Model(data)
    #model = TinyCNN.Model(data)
    model.train()
    model.summary()
    model.test()
    #plot_model(model.model, to_file='TinyCNN.png')


if __name__ == "__main__":
    main()