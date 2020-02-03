from __future__ import print_function, division
from builtins import range 

import numpy as np 
import tensorflow as tf 
import self_q_learning 
tf.compat.v1.disable_eager_execution()

class SGDRegressor: 
    def __init__(self, D):
        print("What's up Tensorflow")
        lr = 0.1 

        ## Create inputs, targets and parameters
        self.w = tf.Variable(tf.random.normal(shape = (D, 1)), name='w')
        self.X = tf.compat.v1.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='Y')

        ## make prediction and cost     
        Y_hat = tf.reshape(tf.matmul(self.X, self.w), [-1])
        delta = self.Y - Y_hat
        cost = tf.reduce_sum(delta*delta)

        #operations to be called later
        self.train_op = tf.compat.v1.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_op = Y_hat

        ## start the session and initialize parameters
        init = tf.compat.v1.global_variables_initializer()
        self.session = tf.compat.v1.InteractiveSession()
        self.session.run(init)

    def partial_fit(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X: X})

if __name__ == '__main__':
    self_q_learning.SGDRegressor = SGDRegressor
    self_q_learning.main()
