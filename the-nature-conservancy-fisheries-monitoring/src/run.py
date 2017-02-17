
from util import *
import tensorflow as tf
import trainable_models.tensorflow_vgg.vgg19_trainable as vgg19
import numpy as np
import math


class Predictor:

    def __init__(self):

        self.graph = tf.Graph()
        
        with self.graph.as_default():

            self.sess = tf.Session()

            vgg = vgg19.Vgg19('src/trained_models/vgg_trained.npy')
            
            self.image_path = tf.placeholder(tf.string)
            
            image = tf.read_file(self.image_path)
            image = tf.image.decode_jpeg(image)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize_images(image, [224, 224])
            image = tf.reshape(image, [1,224,224,3])

            vgg.build(image)

            self.predictor = vgg.prob

            self.sess.run(tf.global_variables_initializer())

    def predict(self, image_path):

        with self.graph.as_default():
            prediction = self.sess.run(
                self.predictor,
                feed_dict={self.image_path: image_path}
            )
            return np.argmax(prediction)
