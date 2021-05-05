# B657_Corum
B657 Final Project for Andrew Corum

To test this code, you will need TensorFlow installed on your machine. GPU acceleration (ie CuDNN) is recommended.

From the top directory, run `python3 main.py` to train and test a few pre-defind models (3 different RandWire models,
AlexNet, TinyCNN, and HandmadeCNN).

To generate new random graphs, run `python3 Graphs/graphs.py`. The new graphs should be displayed in a window and saved to the 
`Graphs/SavedGraphs/` directory.

## General code-related references
* TensorFlow Docs: https://www.tensorflow.org/
* CUDA install/setup docs: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/
* Simple CNN for MNIST dataset: https://linux-blog.anracom.com/2020/05/31/a-simple-cnn-for-the-mnist-datasets-ii-building-the-cnn-with-keras-and-a-first-test/
* Code for AlexNet: https://towardsdatascience.com/alexnet-8b05c5eb88d4
* PyTorch implementation of RandWire: https://github.com/seungwonpark/RandWireNN/tree/0850008e9204cef5fcb1fe508d4c99576b37f995

**Additional references can be found** in the final report and specifically mentioned throughout the code.