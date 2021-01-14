import tensorflow as tf 
import nengo_dl
print(tf.__version__)
print(nengo_dl.__version__)
print(tf.reduce_sum(tf.random.normal([1000,1000])))
