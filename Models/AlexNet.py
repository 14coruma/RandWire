import tensorflow as tf

class Model():
    '''
    AlexNet Model
    Code adapted from https://towardsdatascience.com/alexnet-8b05c5eb88d4
    Which interprets paper:
        Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton.
        Imagenet classification with deepconvolutional neural networks.
        ACM, 60(6):84–90, May 2017.
    '''
    def __init__(self, data):
        self.load_data(data)
        self.reshape_data()
        self.build_model()

    def load_data(self, data):
        self.X_train, self.X_valid, self.X_test = data["X_train"], data["X_valid"], data["X_test"]
        self.y_train, self.y_valid, self.y_test = data["y_train"], data["y_valid"], data["y_test"]
        
    def reshape_data(self):
        # BEGIN: Code from https://towardsdatascience.com/alexnet-8b05c5eb88d4
        self.X_train = tf.map_fn(lambda i: tf.stack([i]*3, axis=-1), self.X_train).numpy()
        self.X_valid = tf.map_fn(lambda i: tf.stack([i]*3, axis=-1), self.X_valid).numpy()
        self.X_test = tf.map_fn(lambda i: tf.stack([i]*3, axis=-1), self.X_test).numpy()

        self.X_train = tf.image.resize(self.X_train, [224, 224]).numpy()
        self.X_valid = tf.image.resize(self.X_valid, [224, 224]).numpy()
        self.X_test = tf.image.resize(self.X_test, [224, 224]).numpy()

        self.X_train = self.X_train.reshape(len(self.X_train), 224, 224, 3)
        self.X_train = self.X_train / 255.0
        self.X_valid = self.X_valid.reshape(len(self.X_valid), 224, 224, 3)
        self.X_valid = self.X_valid / 255.0
        self.X_test = self.X_test.reshape(len(self.X_test), 224, 224, 3)
        self.X_test = self.X_test / 255.0
        # END: Code from https://towardsdatascience.com/alexnet-8b05c5eb88d4
        
    def build_model(self):
        # BEGIN: Code adapted from https://towardsdatascience.com/alexnet-8b05c5eb88d4
        self.model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu',
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    input_shape=(224, 224, 3)),
            
                tf.keras.layers.MaxPooling2D(3, strides=2),
                
                tf.keras.layers.Conv2D(256, (5, 5), activation='relu',
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    bias_initializer='ones'),
                
                tf.keras.layers.MaxPooling2D(3, strides=2),
                
                tf.keras.layers.Conv2D(384, (3, 3), activation='relu',
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
                
                tf.keras.layers.Conv2D(384, (3, 3), activation='relu',
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    bias_initializer='ones'),
                
                tf.keras.layers.Conv2D(384, (3, 3), activation='relu',
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    bias_initializer='ones'),
                
                tf.keras.layers.MaxPooling2D(3, strides=2),
                    
                tf.keras.layers.Flatten(),
                
                tf.keras.layers.Dense(4096,
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    bias_initializer='ones'),
                
                tf.keras.layers.Dropout(0.5),
                
                tf.keras.layers.Dense(4096,
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    bias_initializer='ones'),
                
                tf.keras.layers.Dropout(0.5),
                
                tf.keras.layers.Dense(10, activation='softmax',
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            ])

        #optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=0.0001)
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        # END: Code adapted from https://towardsdatascience.com/alexnet-8b05c5eb88d4

    def train(self, epochs=100, batch_size=64):
        # BEGIN: Code from https://towardsdatascience.com/alexnet-8b05c5eb88d4
        #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', \
        #                                                factor=0.1, patience=1, \
        #                        min_lr=0.00001)


        return self.model.fit(self.X_train, self.y_train, batch_size=batch_size,
            validation_data=(self.X_valid, self.y_valid),
            epochs=epochs)
        # END: Code from https://towardsdatascience.com/alexnet-8b05c5eb88d4

    def test(self):
        res = self.model.evaluate(self.X_test, self.y_test, batch_size=64)
        print("Test loss: {}".format(res[0]))
        print("Test accuracy: {}".format(res[1]))
    
    def summary(self):
        self.model.summary()
