import tensorflow as tf 
import numpy as np 
import pandas as pd

tf.reset_default_graph() 
train = pd.read_csv('C:/Users/Python/Desktop/simple_train.csv', header=None) 
tr = train.values 
def new_sample(A): 
    d = len(A[0,:]) 
    e = np.random.choice(d-10) 
    return A[:,e:(e+10)].transpose().reshape(1,10,4), A[:,(e+1):(e+11)].transpose().reshape(1,10,4)

n_neurons = 100 
n_inputs = 4 
n_outputs = 4 
n_steps = 10
learning_rate = 0.001
 
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) 
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation=tf.nn.relu) 
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1,n_neurons]) 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs) 
logits = tf.reshape(stacked_outputs,[-1,n_steps, n_outputs])

loss = tf.losses.softmax_cross_entropy(y,logits)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) 
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_iterations = 100 
batchsize = 1

with tf.Session() as sess: 
    init.run() 
    for iteration in range(n_iterations): 
        X_batch, y_batch = new_sample(tr) 
        sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})
    X_new, y_new = new_sample(tr) 
    pred_logits = sess.run(logits, feed_dict = {X:X_new}) 
    y_pred = tf.nn.softmax(pred_logits) 
    print(X_new) 
    g = y_pred.eval()*100
    print(g.astype(int))