import tensorflow as tf

"""
The build step
As noted, it's convenient in many cases to wait to create variables until you are sure of the input shape.

Keras layers come with an extra lifecycle step that allows you more flexibility in how you define your layers. This is defined in the build function.

build is called exactly once, and it is called with the shape of the input. It's usually used to create variables (weights).

You can rewrite MyDense layer above to be flexible to the size of its inputs:
"""


class FlexibleDense(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.w = tf.Variable(
            tf.random.normal([input_shape[-1], self.out_features]), name='w')
        self.b = tf.Variable(tf.zeros([self.out_features]), name='b')

    def call(self, inputs):  # Defines the computation from inputs to outputs
        return tf.matmul(inputs, self.w) + self.b


if __name__ == '__main__':
    with tf.device('GPU:0'):
        # Create the instance of the layer
        flexible_dense = FlexibleDense(out_features=3)

        print(flexible_dense.variables)

        # Call it, with predictably random results
        print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])))

        print(flexible_dense.variables)

        try:
            print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0, 2.0]])))
        except tf.errors.InvalidArgumentError as e:
            print("Failed:", e)
