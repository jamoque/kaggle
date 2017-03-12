"""
Run this file to train a VGG 19 Network

Assumptions:

Example usage:

Other comments:
"""

import tensorflow as tf
import trainable_models.tensorflow_vgg.vgg19_trainable as vgg19
import time, os
import config


def evaluate(sess, evaluator, data_size, images, labels, train_mode,
         image_batch, label_batch):
    """
    Runs an evaluation against a full epoch of data
    """
    num_correct = 0
    steps_per_epoch = data_size // config.batch_size
    num_examples = steps_per_epoch * config.batch_size

    for step in xrange(steps_per_epoch):
        np_images, np_labels = sess.run([image_batch, label_batch])
        num_correct += sess.run(evaluator, feed_dict={
                images: np_images,
                labels: np_labels,
                train_mode: False
        })

    precision = float(num_correct) / num_examples

    msg = '\tExamples: {} Correct: {} Precision: {}'.format(
        num_examples,
        num_correct,
        precision
    )
    log(msg, logfile)

def make_batches(label_file, batch_size, num_epochs): 
    images, labels = read_labeled_image_list(label_file)

    input_queue = tf.train.slice_input_producer(
        [images, labels],
        num_epochs=num_epochs,
        shuffle=True
    )

    image, label = read_images_from_disk(input_queue)
    image = tf.image.convert_image_dtype(image, tf.float32)
    shape = None
    try:
        image = tf.image.resize_images(image, [224, 224])
    except:
        image = tf.image.resize_images(image, 224, 224)
    shape = (224, 224, 3)

    image_batch, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        shapes=[shape, ()],
        capacity=3*batch_size
    )

    return image_batch, label_batch, len(images)

def read_labeled_image_list(image_list_file):
    """
    Reads a .txt file containing paths and labels
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filename = config.data_path + filename
        filenames.append(filename)
        labels.append(config.label_to_int(label))
    f.close()

    return filenames, labels

def read_images_from_disk(input_queue):
    """
    Reads a single file from the disk; the filename is return by
    a queue which stores image filenames and their labels
    """
    label = input_queue[1]
    filename = input_queue[0]
    file_contents = tf.read_file(filename)
    example = tf.image.decode_jpeg(file_contents, channels=3)
    return example, label

def log(msg, logfile=None):
    if logfile:
        logfile.write(msg + '\n')
        logfile.flush()
    print msg


# NOTE: For full description of model parameters, see config.py
param_string = "../out/{}.lr.{}.eps.{}.dr.{}.reg.{}.txt".format(
    time.strftime("%m.%d.%Y.%H.%M"),
    config.learning_rate,
    config.epsilon,
    config.dropout,
    config.reg
)
logfile = open(param_string, 'a')

# create mini-batches for training set and test set
# batch info: https://arxiv.org/pdf/1502.03167v3.pdf
train_image_batch, train_label_batch, train_batch_size = make_batches(
    config.training_label_file_path,
    config.batch_size,
    num_epochs=None,
)

test_image_batch, test_label_batch, test_batch_size = make_batches(
    config.test_labels_file_path,
    config.batch_size,
    num_epochs=None,
)

# Placeholders for the images, labels, and training mode
images_ = tf.placeholder(tf.float32, shape=(config.batch_size, 224, 224, 3))
labels_ = tf.placeholder(tf.int32, shape=(config.batch_size))
train_mode_ = tf.placeholder(tf.bool)

# Initialize VGG Network
# Passing the path for the .npy file where we will save the model
# VGG Repo from: TODO - add vgg tensorflow here
vgg = vgg19.Vgg19(
    './trained_models/pretrained_vgg19.npy',
    dropout=config.dropout,
    l2_reg=config.reg,
    num_classes=config.num_classes
)
vgg.build(images_, train_mode_)

# Initialize session
sess = tf.Session()

# Using cross-entropy loss here; might make more sense to try another
logits = vgg.fc8
labels = tf.to_int64(labels_)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits,
    labels=labels,
    name='xentropy'
)

# Consider the cross-entropy loss + vgg regularization loss
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean') + vgg.reg_loss

# Compare the logits to the labels during evaluation
output_comparator = tf.nn.in_top_k(logits, labels_, 1)

# Compute the number of correct entries
eval_comparison = tf.reduce_sum(tf.cast(output_comparator, tf.int32))

# Track the global step count
global_step = tf.Variable(0, name='global_step', trainable=False)

# ADAM Optimizer for gradient descent
optimizer = tf.train.AdamOptimizer(
    learning_rate=config.learning_rate,
    epsilon=config.epsilon
)

# Each training step uses the optimizer to apply the
# gradients that minimize the loss
train_op = optimizer.minimize(loss, global_step=global_step)

# Ssaver for maintaining training checkpoints
saver = tf.train.Saver()

# Begin training
duration = 0.0
cumulative_loss = 0.0
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

log("Beginning Training", logfile)

for step in xrange(config.max_steps):
    start_time = time.time()
    np_images, np_labels = sess.run([train_image_batch, train_label_batch])

    train_feed_dict = {
        images_: np_images,
        labels_: np_labels,
        train_mode_: True
    }

    _, loss_value = sess.run([train_op, loss], feed_dict=train_feed_dict)
    cumulative_loss += loss_value
    duration += time.time() - start_time

    # Record loss every 100 training steps
    if step % config.output_frequency == 0:
        cumulative_loss /= config.output_frequency
        msg = 'Step {}: loss = {} ({} sec)'.format(
            step,
            cumulative_loss,
            duration
        )
        log(msg, logfile)
        
        duration = 0.0
        cumulative_loss = 0.0

    # Save a checkpoint and evaluate the model
    if (step + 1) % config.test_and_save_frequency == 0:
        checkpoint_file = os.path.join(config.train_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=step)

        # Evaluate against the training set
        log('Training Data Eval:', logfile)
        evaluate(
            sess=sess,
            evaluator=eval_comparison,
            data_size=train_batch_size,
            images=images_,
            labels=labels_,
            train_mode=train_mode_,
            image_batch=train_image_batch,
            label_batch=train_label_batch,
        )

        # Evaluate against the validation set.
        log('Validation Data Eval:', logfile)
        evaluate(
            sess=sess,
            evaluator=eval_comparison,
            data_size=test_batch_size,
            images=images_,
            labels=labels_,
            train_mode=train_mode_,
            image_batch=test_image_batch,
            label_batch=test_label_batch
        )
        
        # Save the checkpoint file
        vgg.save_npy(sess, 'trained-step-{}.npy').format(step)

        log('\n', logfile)

# vgg.save_npy() save the model
vgg.save_npy(sess, 'trained-step-{}.npy').format(step)
