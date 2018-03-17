""" 
# -*- coding: utf-8 -*-

Created on Sun May 5 09:57:22 2013

@author: Sagar


Phase 1 code to test the theory of generating images by GANs.
This is a simple feed forward network with a single hidden layer.
Since this network is only to be used for testing with mnist, 
network dimensions are fixed to 784(28x28 for mnist), 128 and 1 for discriminator.
64, 128 and 784 for the generator.




"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

mb_size = 32	#batch size
X_dim = 784	#first layer of dicriminator, and last layer of generator	
z_dim = 64	#noise layer of generator
h_dim = 128	#hidden layer
lr = 1e-3	#learning rate
d_steps = 3	#discriminator steps

mnist = input_data.read_data_sets('data/mnist', one_hot=True)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')     #since mnist is greyscale

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def log(x):
    return tf.log(x + 1e-8)


X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]
theta_D = [D_W1, D_W2, D_b1, D_b2]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def G(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def D(X):
    D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


G_sample = G(z)

D_real = D(X)
D_fake = D(G_sample)

D_target = 1./mb_size
G_target = 1./(mb_size*2)

Z = tf.reduce_sum(tf.exp(-D_real)) + tf.reduce_sum(tf.exp(-D_fake))

D_loss = tf.reduce_sum(D_target * D_real) + log(Z)
G_loss = tf.reduce_sum(G_target * D_real) + tf.reduce_sum(G_target * D_fake) + log(Z)

D_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)
    z_mb = sample_z(mb_size, z_dim)

    _, D_loss_curr = sess.run(
        [D_solver, D_loss], feed_dict={X: X_mb, z: z_mb}
    )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss], feed_dict={X: X_mb, z: z_mb}
    )

	#save an image every 1000th iteration
    if it % 1000 == 0:
        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))

        samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
