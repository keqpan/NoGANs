import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow.contrib.slim as slim #slim = tf.contrib.slim
import tensorflow.contrib.slim.nets



from tensorflow.contrib.framework.python.ops import add_arg_scope

import sys, os
sys.path.append('../models/research/slim/')
from datasets import dataset_factory, dataset_utils

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_integer('num_epochs', 50, 'Num of epochs to train.')
flags.DEFINE_integer('val_every_n_epoch', 5, 'Validation every n epochs.')
flags.DEFINE_float("lr", 0.0001, "Learning rate [0.0001].")
flags.DEFINE_string('log_dir', './train_log/',
                    'Directory with the log data.')
FLAGS = flags.FLAGS


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


reader = tf.TFRecordReader

keys_to_features = {
    'image/encoded': tf.FixedLenFeature(
        (), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature(
        (), tf.string, default_value='jpeg'),
    'image/image_id': tf.FixedLenFeature(
        (), dtype=tf.int64, default_value=-1),
    'image/prototype_id': tf.FixedLenFeature(
        (), dtype=tf.int64, default_value=-1),
}

items_to_handlers = {
    'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    'image_id': slim.tfexample_decoder.Tensor('image/image_id'),
    'prototype_id': slim.tfexample_decoder.Tensor('image/prototype_id'),
}

decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

train_dataset_size = 26650
train_dataset = slim.dataset.Dataset(
      data_sources=['train_w_ids.tfrecord'],
      reader=reader,
      decoder=decoder,
      num_samples=train_dataset_size,
      items_to_descriptions='fake descr')

batch_size = FLAGS.batch_size 
# n_epochs = FLAGS.num_epochs
epoch_size = int(train_dataset_size/batch_size)
learning_rate = FLAGS.lr
validation_every_n_step = FLAGS.val_every_n_epoch * epoch_size

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

desired_size = (224, 224)

im = Image.open('proto_0.jpg')
im = im.resize(desired_size, Image.ANTIALIAS)
im = im.convert('RGB')
proto0 = np.asarray(im)

im = Image.open('proto_1.jpg')
im = im.resize(desired_size, Image.ANTIALIAS)
im = im.convert('RGB')
proto1 = np.asarray(im)

proto0 = proto0.astype(np.float32)
proto1 = proto1.astype(np.float32)
proto0 = ((proto0/255)-0.5)*2
proto1 = ((proto1/255)-0.5)*2

from model import vgg_19_encoder, vgg_19_decoder

def train(autoencoder, n_epochs):
    img_width = 224
    img_height = 224
    noise_dim = 1

    optimizer= tf.train.AdamOptimizer(learning_rate)

    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            if autoencoder:
                z = tf.zeros(shape=[train_dataset_size, 28, 28, noise_dim], name="Z")
            else:
                z = tf.get_variable("Z", shape=[train_dataset_size, 28, 28, noise_dim], dtype=tf.float32,  initializer=tf.zeros_initializer )   

        with tf.device('/cpu:0'):
            global_step = tf.train.get_or_create_global_step()
            provider_train = slim.dataset_data_provider.DatasetDataProvider(
                  train_dataset,
                  num_readers=1,
                  common_queue_capacity=20 * batch_size,
                  common_queue_min=10 * batch_size)
            inputs, img_id, proto_id = provider_train.get(['image', 'image_id', 'prototype_id'])
            inputs = tf.cast(inputs, tf.float32)
            inputs = tf.reshape(inputs, tf.stack([img_height, img_width, 3]))
            inputs = tf.divide(inputs, 255)
            inputs = 2*(inputs - 0.5)
            inputs = tf.expand_dims(inputs, 0)
            img_id = tf.expand_dims(img_id, 0)
            proto_id = tf.expand_dims(proto_id, 0)

            real_imgs, img_ids, proto_ids = tf.train.batch([inputs, img_id, proto_id], batch_size, enqueue_many=True, num_threads=4, capacity=5 * batch_size)

            batch_queue = slim.prefetch_queue.prefetch_queue([real_imgs, img_ids, proto_ids], capacity=2)      

        with tf.device('/gpu:0'):
            proto_0 = tf.tile(tf.expand_dims(proto0,0), [batch_size,1,1,1])
            proto_1 = tf.tile(tf.expand_dims(proto1,0), [batch_size,1,1,1])
            real_imgs, img_ids, proto_ids = batch_queue.dequeue()
            prototypes = tf.where(tf.cast(proto_ids, tf.bool), x=proto_1, y=proto_0)
            z_curr = tf.gather(z, indices=img_ids)
#             z_curr = tf.tile(z_curr, [1, 28, 1, 1])  

            vgg_encodings, p1, p2, p3 = vgg_19_encoder(real_imgs)
            encodings = slim.conv2d(tf.concat([vgg_encodings, z_curr], axis=3), 512, [3, 3], scope='1x1_conv')
            gen_outputs = tf.tanh(vgg_19_decoder(encodings, p1, p2, p3))

            auenc_loss_L1 = 100*tf.reduce_mean(tf.abs(real_imgs - gen_outputs))

            model_vars = tf.trainable_variables()
            variables_to_train = []
            variables_to_restore = []
            for var in model_vars:
                if autoencoder:
                    if var.op.name.startswith('vgg_19/'):
                        variables_to_restore.append(var)
                else:
                    if var.op.name.startswith('vgg_19/') or var.op.name.startswith('vgg_19_decoder/'):
                        variables_to_restore.append(var)
                    if var.op.name.startswith('Z') or var.op.name.startswith('1x1_conv'):
                        variables_to_train.append(var)
            if autoencoder:
                variables_to_train = tf.trainable_variables()
    #         print("Variables to restore:", variables_to_restore)
            print("Variables to train:", variables_to_train)
            train_tensor = slim.learning.create_train_op(auenc_loss_L1, optimizer, variables_to_train=variables_to_train)
        if autoencoder:
            train_log_dir = os.path.join(FLAGS.log_dir, "auenc/")
        else:
            train_log_dir = os.path.join(FLAGS.log_dir, "w_noise/")
        
        summaries = set()
        summaries.add(tf.summary.scalar('loss', auenc_loss_L1))
        img_tensor = tf.concat([tf.gather(real_imgs,[0,1,2], axis=0), tf.gather(gen_outputs,[0,1,2], axis=0)], axis=1)
        summaries.add(tf.summary.image('imgs', img_tensor))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        if autoencoder:
             init_fn = slim.assign_from_checkpoint_fn('../vgg_19/vgg_19.ckpt', variables_to_restore, ignore_missing_vars=True)
        else:
             init_fn = slim.assign_from_checkpoint_fn(os.path.join(FLAGS.log_dir, "auenc/"), variables_to_restore, ignore_missing_vars=True)

        slim.learning.train(
            train_tensor, 
            train_log_dir,
            summary_op=summary_op,
            log_every_n_steps = epoch_size,
            session_config = config,
            save_interval_secs = 600,
            save_summaries_secs = 120,
            number_of_steps = epoch_size*n_epochs,
            init_fn = init_fn,
            global_step = global_step
        )


def main(args):
    train(autoencoder=True, n_epochs=10)
    train(autoencoder=False, n_epochs=FLAGS.num_epochs)


if __name__ == '__main__':
    tf.app.run()
