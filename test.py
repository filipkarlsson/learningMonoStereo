'''
HelloWorld example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import custom_losses
import bootstrap
from PIL import Image
import numpy as np
import vis


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pickle
import helpers



#vis.vis_image_pair(pair, 'Gopher')

# Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!')

a = 0.1*tf.random_normal((12,48,64,1), stddev=0.1)
b = 0.7*tf.ones((12,6))
z = 0*a
n  = a* float('inf')

# true inf

#c = custom_losses.euclidean_with_gradient_loss(a, z)
#c = custom_losses.flow_scale_invariant_loss(a,b)
# Start tf session
c = helpers.flow_from_depth(a, b)
sess = tf.Session()
print(c.shape)

# Run the op
print(sess.run(c))


#bootstrap_net = bootstrap.bootstrap_net()
#print(bootstrap_net.outputs)
