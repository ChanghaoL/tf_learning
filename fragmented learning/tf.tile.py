import tensorflow as tf

# tile(
#     input,     #输入
#     multiples,  #同一维度上复制的次数
#     name=None
# )
# 在对应维度上复制

if __name__ == "__main__":
    a = tf.constant([[1, 2, 3], [4, 5, 6]], tf.int32)
    print(a.shape)
    b = tf.constant([1, 2], tf.int32)
    tf.print(tf.tile(a, b))
    c = tf.constant([2, 1], tf.int32)
    tf.print(tf.tile(a, c))
    d = tf.constant([2, 2], tf.int32)
    tf.print(tf.tile(a, d))
