# Introduction to graphs and tf.function
本章节参考 [tf官网](https://www.tensorflow.org/guide/intro_to_graphs?hl=en) 编写
## 1.[What are graphs?](https://www.tensorflow.org/guide/intro_to_graphs?hl=en#what_are_graphs)
Graphs are data structures that contain a set of tf.Operation objects, which represent units of computation; and tf.Tensor objects, which represent the units of data that flow between operations. 
```
import tensorflow as tf 
import timeit
from datetime import datetime
```
### Taking advantage of graphs
You create and run a graph in TensorFlow by using tf.function, either as a direct call or as a decorator. tf.function takes a regular function as input and returns a Function. A Function is a Python callable that builds TensorFlow graphs from the Python function. You use a Function in the same way as its Python equivalent.
```buildoutcfg
# Define a Python function.
def a_regular_function(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

# `a_function_that_uses_a_graph` is a TensorFlow `Function`.
a_function_that_uses_a_graph = tf.function(a_regular_function)

# Make some tensors.
x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

orig_value = a_regular_function(x1, y1, b1).numpy()
# Call a `Function` like a Python function.
tf_function_value = a_function_that_uses_a_graph(x1, y1, b1).numpy()
assert(orig_value == tf_function_value)
```
### Non-strict execution

Graph execution only executes the operations necessary to produce the observable effects, which includes:

- The return value of the function
Documented well-known side-effects such as:
Input/output operations, like tf.print
- Debugging operations, such as the assert functions in tf.debugging
Mutations of tf.Variable
This behavior is usually known as "Non-strict execution", and differs from eager execution, which steps through all of the program operations, needed or not.

In particular, runtime error checking does not count as an observable effect. If an operation is skipped because it is unnecessary, it cannot raise any runtime errors.

In the following example, the "unnecessary" operation tf.gather is skipped during graph execution, so the runtime error InvalidArgumentError is not raised as it would be in eager execution. Do not rely on an error being raised while executing a graph.

```buildoutcfg
def unused_return_eager(x):
  # Get index 1 will fail when `len(x) == 1`
  tf.gather(x, [1]) # unused 
  return x

try:
  print(unused_return_eager(tf.constant([0.0])))
except tf.errors.InvalidArgumentError as e:
  # All operations are run during eager execution so an error is raised.
  print(f'{type(e).__name__}: {e}')
```
