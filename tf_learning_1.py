"""
Tensorflow works on principle of dataflow graphs. To perform some computation there are two steps:
    1.Represent the computation as a graph
    2.Execute the graph

Representation: Like any directed graph a Tensorflw graph consists of nodes and directional edges.

Node: A Node is also called an OP(stands for operation). An node can have multiple incomint edges but outgoing edge.

Edge: indicate incoming or outgoing data from a Node.

Whenever we say data we mena an n-dimensinal vector known as Tensor. A Tensor has three properties: Rank, Shape and Type:
    Rank means number of dimnesions of the Tensor(a cube or box has rank 3).
    Shape menas values of those dimensions(box can have shape 1x1x1 or 2x5x7).
    Type means datatype in each coordinate of Tensor

Execution: Even though a graph is constructed it is still an abstract entity. No computation actually occurs until we run it. To run a graph, we need to allocate CPU/GPU resource to OPs inside the graph. This is done using Tensorflow Sessions. Steps are:
    1.Create a new session.
    2.Run any Op inside the Graph. Usually we run the final Op where we expect the output of our computation.

An incoming edge on an Op is like a dependecy for data on another Op. Thus when we run any Op, all incoming edges on it are traced and the ops on other side are also run.

Note: Special nodes called playing role of data source or sink are also possible. For example you can have an Op which gives a constant value thus no incoming edges(refer value 'matrix1' in the next example below) and simiarly Op with no ougoing edges where results are collected(refer value 'product' in the example below).

"""

import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

