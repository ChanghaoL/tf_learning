import tensorflow as tf


def get_tensor():
    x = tf.random.uniform((5, 4))
    ind = tf.where(x > 0.5)
    y = tf.gather_nd(x, ind)
    return x, ind, y


x, ind, y = get_tensor()

tf.print(x)
tf.print(ind)
tf.print(tf.reshape(y, [-1, 1]))
print(x.shape)
print(tf.reshape(y, [-1, 1]).shape)
