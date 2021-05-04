from Datasets import MNIST
from Models import AlexNet, TinyCNN, RandWire, HandmadeCNN

from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import numpy as np

def main():
    # Get dataset
    data = MNIST.get_data(n=6000, m=1000)

    names = ['AlexNet', 'TinyCNN', 'HandmadeCNN', 'RandWire, WS(4, .75)']
    for name in names:
        model = None
        # Build the model
        if name == 'AlexNet': model = AlexNet.Model(data)
        elif name == 'TinyCNN': model = TinyCNN.Model(data)
        elif name == 'HandmadeCNN': model = HandmadeCNN.Model(data)
        elif name == 'RandWire, WS(4, .75)': model = RandWire.Model(data)

        # Train & test model
        history = model.train(epochs=10)
        model.test()

        # Plot learning curves 
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.title("{} Accuracy".format(name))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(10),  acc, label="Training Accuracy")
        plt.plot(range(10), val_acc, label="Validation Accuracy")
        plt.legend()
        plt.show()
        
        plt.title("{} Loss".format(name))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(range(10),  loss, label="Training Loss")
        plt.plot(range(10), val_loss, label="Validation Loss")
        plt.legend()
        plt.show()

        # Destroy model
        del model

if __name__ == "__main__":
    main()