import datetime

import tensorflow as tf

"""
Overview of tf module
"""


class SimpleModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name="train_me")
        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")

    def __call__(self, x):
        return self.a_variable * x + self.non_trainable_variable


simple_module = SimpleModule(name="simple")

simple_module(tf.constant(5.0))

"""Note: tf.Module is the base class for both tf.keras.layers.Layer and tf.keras.Model, so everything you come across 
here also applies in Keras. For historical compatibility reasons Keras layers do not collect variables from modules, 
so your models should use only modules or only Keras layers. However, the methods shown below for inspecting 
variables are the same in either case. """

"""
An example of using model
"""


class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


class SequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


# You have made a model!
my_model = SequentialModule(name="the_model")

# Call it, with random results
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))

"""
Flexible variable size
"""


class FlexibleDenseModule(tf.Module):
    # Note: No need for `in_features`
    def __init__(self, out_features, name=None):
        super().__init__(name=name)
        self.is_built = False
        self.out_features = out_features

    def __call__(self, x):
        # Create variables on first call.
        if not self.is_built:
            self.w = tf.Variable(
                tf.random.normal([x.shape[-1], self.out_features]), name='w')
            self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
            self.is_built = True

        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


# Used in a module
class MySequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = FlexibleDenseModule(out_features=3)
        self.dense_2 = FlexibleDenseModule(out_features=2)

    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


my_model = MySequentialModule(name="the_model")
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))

"""
Saving weights
You can save a tf.Module as both a checkpoint and a SavedModel.

Checkpoints are just the weights (that is, the values of the set of variables inside the module and its submodules):
只是权重值
"""

chkp_path = "my_checkpoint"
checkpoint = tf.train.Checkpoint(model=my_model)
checkpoint.write(chkp_path)

"""Checkpoints consist of two kinds of files: the data itself and an index file for metadata. The index file keeps 
track of what is actually saved and the numbering of checkpoints, while the checkpoint data contains the variable 
values and their attribute lookup paths. """

"""https://www.tensorflow.org/guide/intro_to_modules?hl=en#saving_functions 
TensorFlow can run models without the 
original Python objects, as demonstrated by TensorFlow Serving and TensorFlow Lite, even when you download a trained 
model from TensorFlow Hub. 

TensorFlow needs to know how to do the computations described in Python, but without the original code. To do this, 
you can make a graph, which is described in the Introduction to graphs and functions guide. 

This graph contains operations, or ops, that implement the function.

You can define a graph in the model above by adding the @tf.function decorator to indicate that this code should run as a graph.
"""


class MySequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    @tf.function
    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


# You have made a model with a graph!
my_model = MySequentialModule(name="the_model")

# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/func/%s" % stamp
writer = tf.summary.create_file_writer(logdir)

# Create a new model to get a fresh trace
# Otherwise the summary will not see the graph.
new_model = MySequentialModule()

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True)
tf.profiler.experimental.start(logdir)
# Call only one tf.function when tracing.
print(new_model(tf.constant([[2.0, 2.0, 2.0]])))
with writer.as_default():
    tf.summary.trace_export(
        name="my_func_trace",
        step=0,
        profiler_outdir=logdir)

"""
tf.keras.layers.Layer is the base class of all Keras layers, and it inherits from tf.Module.

You can convert a module into a Keras layer just by swapping out the parent and then changing __call__ to call:
更改call
"""


class MyDense(tf.keras.layers.Layer):
    # Adding **kwargs to support base Keras layer arguments
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(**kwargs)

        # This will soon move to the build step; see below
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def call(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


simple_layer = MyDense(name="simple", in_features=3, out_features=3)
