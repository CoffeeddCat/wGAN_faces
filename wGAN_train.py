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
        self.x_dim = 64*64
        self.z_dim = 100
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')



datapaths = glob.glob('./faces/*.jpg')
data = dataset(image_paths=datapaths,
               batch_size=64,
               shape=[96,96,3],
               preprocess_fn=preprocess_fn
               )

G = generator()
D = discriminator()
gan = wGAN(generator, discriminator, data)
