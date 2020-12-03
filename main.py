import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
a=tf.constant(2)
b=tf.constant(3)
with tf.compat.v1.Session() as sess:
    print("a:%i" % sess.run(a),"b:%i" % sess.run(b))
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constant:%i" % sess.run(a*b))

a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)
add=tf.add(a,b)
mul=tf.multiply(a,b)
with tf.Session() as sess:
    print("Addition with variables: %i" % sess.run(add,feed_dict={a:2,b:3}))
    print("Multiplication with variables: %i" % sess.run(mul,feed_dict={a:2,b:3}))

with tf.Session() as sess:
    matrix1=tf.constant([[3.,3.]])
    matrix2=tf.constant([[2.],[2.]])
    product=tf.matmul(matrix1,matrix2)
    result=sess.run(product)
    print(result)