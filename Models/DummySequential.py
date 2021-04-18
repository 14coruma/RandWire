import tensorflow as tf
from datetime import datetime

class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

class MySequentialModel(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    @tf.function
    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)

if __name__ == "__main__":
    # Set up logging
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/func/%s" % stamp
    writer = tf.summary.create_file_writer(logdir)

    # Create new model
    model = MySequentialModel(name="simple")

    tf.summary.trace_on(graph=True)
    tf.profiler.experimental.start(logdir)
    print(model(tf.constant([[2.0, 2.0, 2.0]])))
    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir
        )