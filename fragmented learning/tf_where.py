import tensorflow as tf

if __name__ == "__main__":
    a = tf.constant([[1, 0, 1], [1, 1, 1]], tf.int32)
    b = tf.constant([[0, 0, 1], [0, 1, 1]], tf.int32)
    tf.print(tf.where(tf.equal(a, 1) & tf.equal(b, 1), 1, 0))
