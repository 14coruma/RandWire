import tensorflow as tf
from tensorflow.keras import layers, models, losses

class Model():
    '''
    Code adapted from https://www.tensorflow.org/tutorials/images/cnn 
    '''
    def __init__(self, data):
        self.load_data(data)
        self.reshape_data()
        self.build_model()

    def load_data(self, data):
        self.X_train, self.X_valid, self.X_test = data["X_train"], data["X_valid"], data["X_test"]
        self.y_train, self.y_valid, self.y_test = data["y_train"], data["y_valid"], data["y_test"]
        
    def reshape_data(self):
        self.X_train = tf.map_fn(lambda i: tf.stack([i]*3, axis=-1), self.X_train).numpy()
        self.X_valid = tf.map_fn(lambda i: tf.stack([i]*3, axis=-1), self.X_valid).numpy()
        self.X_test = tf.map_fn(lambda i: tf.stack([i]*3, axis=-1), self.X_test).numpy()

        self.X_train = tf.image.resize(self.X_train, [32, 32]).numpy()
        self.X_valid = tf.image.resize(self.X_valid, [32, 32]).numpy()
        self.X_test = tf.image.resize(self.X_test, [32, 32]).numpy()

        self.X_train = self.X_train.reshape(800, 32, 32, 3)
        self.X_train = self.X_train / 255.0
        self.X_valid = self.X_valid.reshape(200, 32, 32, 3)
        self.X_valid = self.X_valid / 255.0
        self.X_test = self.X_test.reshape(100, 32, 32, 3)
        self.X_test = self.X_test / 255.0
        
    def build_model(self):
        # BEGIN: Code adapted from https://www.tensorflow.org/tutorials/images/cnn
        self.model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ])

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        # END: Code adapted from https://www.tensorflow.org/tutorials/images/cnn

    def train(self):
        self.model.fit(self.X_train, self.y_train, batch_size=128,
            validation_data=(self.X_valid, self.y_valid),
            epochs=90)

    def test(self):
        res = self.model.evaluate(self.X_test, self.y_test, batch_size=128)
        print("Test loss: {}".format(res[0]))
        print("Test accuracy: {}".format(res[1]))
    
    def summary(self):
        self.model.summary()
