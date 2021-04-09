import tensorflow as tf

class Model():
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

        self.X_train = tf.image.resize(self.X_train, [224, 224]).numpy()
        self.X_valid = tf.image.resize(self.X_valid, [224, 224]).numpy()
        self.X_test = tf.image.resize(self.X_test, [224, 224]).numpy()

        self.X_train = self.X_train.reshape(800, 224, 224, 3)
        self.X_train = self.X_train / 255.0
        self.X_valid = self.X_valid.reshape(200, 224, 224, 3)
        self.X_valid = self.X_valid / 255.0
        self.X_test = self.X_test.reshape(100, 224, 224, 3)
        self.X_test = self.X_test / 255.0
        
    def build_model(self):
        self.model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', \
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), \
                    input_shape=(224, 224, 3)),
            
                tf.keras.layers.MaxPooling2D(3, strides=2),
                
                tf.keras.layers.Conv2D(256, (5, 5), activation='relu', \
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), \
                    bias_initializer='ones'),
                
                tf.keras.layers.MaxPooling2D(3, strides=2),
                
                tf.keras.layers.Conv2D(384, (3, 3), activation='relu', \
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
                
                tf.keras.layers.Conv2D(384, (3, 3), activation='relu', \
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), \
                    bias_initializer='ones'),
                
                tf.keras.layers.Conv2D(384, (3, 3), activation='relu', \
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), \
                    bias_initializer='ones'),
                
                tf.keras.layers.MaxPooling2D(3, strides=2),
                    
                tf.keras.layers.Flatten(),
                
                tf.keras.layers.Dense(4096, kernel_initializer=\
                                    tf.random_normal_initializer(mean=0.0, stddev=0.01), \
                            bias_initializer='ones'),
                
                tf.keras.layers.Dropout(0.5),
                
                tf.keras.layers.Dense(4096, kernel_initializer=\
                                    tf.random_normal_initializer(mean=0.0, stddev=0.01), \
                            bias_initializer='ones'),
                
                tf.keras.layers.Dropout(0.5),
                
                tf.keras.layers.Dense(10, activation='softmax', \
                            kernel_initializer= \
                            tf.random_normal_initializer(mean=0.0, stddev=0.01))
            ])

        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9), \
                    loss='categorical_crossentropy', \
                metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(5)])

    def train(self):
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', \
                                                        factor=0.1, patience=1, \
                                min_lr=0.00001)

        self.model.fit(self.X_train, self.y_train, batch_size=128, \
                validation_data=(self.X_valid, self.y_valid), \
            epochs=90, callbacks=[reduce_lr])
    
    def __str__(self):
        return self.model.summary()
