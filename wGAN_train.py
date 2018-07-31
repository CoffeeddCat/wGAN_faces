import glob
import tensorflow as tf
import numpy as np
from utils import dataset
from models import generator, discriminator

def preprocess_fn(img):
    re_size = 64
    img = tf.to_float(tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1  #resize the img
    return img

class wGAN:
    def __init__(self, G, D, dataset):
        self.G = G
        self.D = D
        self.dataset = dataset
        self.x_dim = 96*96
        self.z_dim = 100
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.x_ = self.G(self.z)

        self.d = self.D(self.x, reuse=False)
        self.d_ = self.D(self.x_)

        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

        #regularization
        self.reg = tf.contrib.layers.apply_regularization(
            tf.layers.l1_regularizer(2.5e-5),
            weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        )

        self.g_loss_reg = self.g_loss + self.reg
        self.d_loss_reg = self.d_loss + self.reg

        self.d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5) \
            .minimize(self.d_loss_reg, var_list=self.D.vars)
        self.g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5) \
            .minimize(self.g_loss_reg, var_list=self.G.vars)

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.D.vars]
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, batch_size=64, num_batches=1000000, n_critic=5):
        self.sess.run(tf.global_variables_initializer())
        for t in range(num_batches):
            if t % 500 == 0 or t < 25:
                n_critic = 100

            for _ in range(n_critic):
                bx = self.dataset.batch()
                bz = np.random.normal(size=[batch_size, self.z_dim])
                self.sess.run(self.d_clip)
                self.sess.run(self.d_rmsprop, feed_dict={
                    self.x: bx,
                    self.z: bz
                })

            bz = np.random.normal(size=[batch_size, self.z_dim])
            self.sess.run(self.g_rmsprop, feed_dict={
                self.z: bz
            })

            if t % 100 == 0:
                bx = self.dataset.batch()
                bz = np.random.normal(size=[batch_size, self.z_dim])

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={
                        self.x: bx,
                        self.z: bz
                    }
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={
                        self.z: bx,
                        self.z: bz
                    }
                )
                print('Iter time: %8d, d_loss: %.4f, g_loss: %.4f' % (t, d_loss, g_loss))


datapaths = glob.glob('./faces/*.jpg')
data = dataset(image_paths=datapaths,
               batch_size=64,
               shape=[96,96,3],
               preprocess_fn=preprocess_fn
               )
G = generator()
D = discriminator()
gan = wGAN(G, D, data)
gan.train()
