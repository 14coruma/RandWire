import tensorflow as tf
from datetime import datetime

class Node(tf.Module):
    # Referenced this PyTorch implementation while working:
    # https://github.com/seungwonpark/RandWireNN/blob/0850008e9204cef5fcb1fe508d4c99576b37f995/model/node.py#L8
    def __init__(self, in_degree, in_channel, out_channel, stride, name=None):
        super(Node, self).__init__(name=name)
        self.single = (in_degree == 1)
        if not self.single:
            # Aggregate sum
            self.agg_weight = None #TODO
        self.conv = None #TODO
        self.bn = None #TODO
    
    # Referenced this PyTorch implementation while working:
    # https://github.com/seungwonpark/RandWireNN/blob/0850008e9204cef5fcb1fe508d4c99576b37f995/model/node.py#L8
    def __call__(self, x):
        pass

class DAG(tf.Module):
    # Referenced this PyTorch implementation while working:
    # https://github.com/seungwonpark/RandWireNN/blob/0850008e9204cef5fcb1fe508d4c99576b37f995/model/dag_layer.py
    def __init__(self, in_channel, out_channel, num_nodes, edges, name=None):
        super(DAG, self).__init__(name=name)
        self.num_nodes, self.edges = num_nodes, edges
        self.adjlist = [[] for node in range(num_nodes)]
        self.rev_adjlist = [[] for node in range(num_nodes)]
        self.in_degree = [0 for node in range(num_nodes)]
        self.out_degree = [0 for node in range(num_nodes)]

        # Build list of degrees and adjacencies for the DAG
        # TODO: Maybe do this in the 'graph.py' code instead?
        for start, end in edges:
            self.in_degree[end] += 1
            self.out_degree[start] += 1
            self.adjlist[start].append(end)
            self.rev_adjlist[end].append(start)

        # Determine input/output nodes
        # TODO: Maybe do this in the 'graph.py' code instead?
        self.input_nodes = [node for node in range(num_nodes) if self.in_degree[node] == 0]
        self.output_nodes = [node for node in range(num_nodes) if self.out_degree[node] == 0]
        assert len(self.input_nodes) > 0, '%d' % len(self.input_nodes)
        assert len(self.output_nodes) > 0, '%d' % len(self.output_nodes)
        for node in self.input_nodes:
            assert len(self.rev_adjlist[node]) == 0
            self.rev_adjlist[node].append(-1)

        self.nodes = [
            Node(
                self.in_degree[node],
                in_channel,
                out_channel,
                2 if node in self.input_nodes else 1, #TODO Why is this the stride?
                name=str(node)
            )
            for node in range(num_nodes)]

    # Referenced this PyTorch implementation while working:
    # https://github.com/seungwonpark/RandWireNN/blob/0850008e9204cef5fcb1fe508d4c99576b37f995/model/dag_layer.py
    def __call__(self, x):
        # input x shape: [Batch, Channel, N, M]
        # Place x at position -1, so input nodes grab x values.
        # Remaining outputs should be None, as they are not yet computed
        outputs = [None for node in range(self.num_nodes)] + [x]
        # Queue up input nodes
        queue = self.input_nodes.copy()
        in_degree = self.in_degree.copy()

        # BFS through DAG, evaluating nodes that have their inputs already computed
        while queue:
            node = queue.pop(0)
            # Grab computed inputs (in the outputs list) for current node
            inputs = [outputs[i] for i in self.rev_adjlist[node]]
            inputs = tf.stack(inputs, axis=-1)
            # Comput output for current node
            outputs[node] = self.nodes[node](inputs)
            # Check if any of the next nodes are ready to evaluate
            for next_node in self.adjlist[node]:
                in_degree[next_node] -= 1
                if in_degree[next_node] == 0: queue.append(next_node)

        # Combine all outputs by averaging
        y = [outputs[node] for node in self.output_nodes]
        return tf.math.reduce_mean(tf.stack(y), axis=0)

class RandWire(tf.Module):
    # Referenced this PyTorch implementation while working:
    # https://github.com/seungwonpark/RandWireNN/blob/0850008e9204cef5fcb1fe508d4c99576b37f995/model/model.py
    def __init__(self, name=None):
        super().__init__(name=name)
        self.dag = DAG(self.chn, )
        #self.dag3 = DAG(in_features=3, out_features=3)
        #self.dag4 = DAG(in_features=3, out_features=2)
        #self.dag5 = DAG(in_features=3, out_features=2)

    @tf.function
    def __call__(self, x):
        x = self.dag(x)
        return x

class Model:
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

    def build_model():
        pass

if __name__ == "__main__":
    # Set up logging
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/func/%s" % stamp
    writer = tf.summary.create_file_writer(logdir)

    # Create new model
    m = RandWire(name="simple")

    tf.summary.trace_on(graph=True)
    tf.profiler.experimental.start(logdir)
    print(m(tf.constant([[2.0, 2.0, 2.0]])))
    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir
        )