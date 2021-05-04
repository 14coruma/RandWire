import tensorflow as tf
from tensorflow.keras import layers, models, losses

class Model():
    '''
    Code adapted from https://www.tensorflow.org/tutorials/images/cnn 
    '''
    def __init__(self, data, location='Models/OCR_CNN_Trained'):
        if data is None:
            self.model = models.load_model(location)
        else:
            self.load_data(data)
            self.reshape_data()
            self.build_model()

    def load_data(self, data):
        self.X_train, self.X_valid, self.X_test = data["X_train"], data["X_valid"], data["X_test"]
        self.y_train, self.y_valid, self.y_test = data["y_train"], data["y_valid"], data["y_test"]
        
    def reshape_data(self):
        self.X_train = self.X_train.reshape(len(self.X_train), 28, 28, 1)
        self.X_train = self.X_train / 255.0
        self.X_valid = self.X_valid.reshape(len(self.X_valid), 28, 28, 1)
        self.X_valid = self.X_valid / 255.0
        self.X_test = self.X_test.reshape(len(self.X_test), 28, 28, 1)
        self.X_test = self.X_test / 255.0
        
    def build_model(self):
        # Model started with https://linux-blog.anracom.com/2020/05/31/a-simple-cnn-for-the-mnist-datasets-ii-building-the-cnn-with-keras-and-a-first-test/
        # Then tested and updated for improvements
        self.model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Conv2D(64, (3,3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Conv2D(64, (3,3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Flatten(),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(10, activation='softmax')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=0.0001)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    def train(self, epochs=100, batch_size=2048):
        return self.model.fit(self.X_train, self.y_train, batch_size=batch_size,
            validation_data=(self.X_valid, self.y_valid),
            epochs=epochs)

    def test(self):
        res = self.model.evaluate(self.X_test, self.y_test, batch_size=128)
        print("Test loss: {}".format(res[0]))
        print("Test accuracy: {}".format(res[1]))

    def save(self, location):
        self.model.save(location)

    def summary(self):
        self.model.summary()

if __name__ == "__main__":
    data = MNIST.get_data(n=60000, m=10000)
    model = Model(data)
    model.train()
    model.summary()
    model.test()
    model.save('Models/OCR_CNN_Trained')