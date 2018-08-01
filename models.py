import tensorflow as tf
import tensorflow.contrib as tc

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class discriminator(object):
    def __init__(self):
        self.dim = 64*64
        self.name = 'discriminator'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs,96,96,3])
            conv1 = tf.contrib.layers.conv2d(
                x, 64, [4,4], [2,2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn = tf.identity
            )
            conv1 = leaky_relu(conv1)
            conv2 = tf.contrib.layers.conv2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv2 = leaky_relu(conv2)
            conv2 = tf.contrib.layers.flatten(conv2)
            fc1 = tf.contrib.layers.fully_connected(
                conv2, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            fc2 = tf.contrib.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
            return fc2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class generator(object):
    def __init__(self):
        self.z_dim = 100
        self.x_dim = 64*64
        self.name = 'generator'

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            bs = tf.shape(z)[0]
            fc1 = tc.layers.fully_connected(
                z, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc1 = tc.layers.batch_norm(fc1)
            fc1 = tf.nn.relu(fc1)
            fc2 = tc.layers.fully_connected(
                fc1, 24 * 24 * 128,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc2 = tf.reshape(fc2, tf.stack([bs, 24, 24, 128]))
            fc2 = tc.layers.batch_norm(fc2)
            fc2 = tf.nn.relu(fc2)
            conv1 = tc.layers.conv2d_transpose(
                fc2, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv1 = tc.layers.batch_norm(conv1)
            conv1 = tf.nn.relu(conv1)
            conv2 = tc.layers.conv2d_transpose(
                conv1, 3, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.sigmoid
            )
            conv2 = tf.reshape(conv2, tf.stack([bs, 96*96*3]))
            return conv2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]