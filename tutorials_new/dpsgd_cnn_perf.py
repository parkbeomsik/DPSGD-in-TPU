# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training a CNN on MNIST with Keras and the DP SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import sys
import os
from datetime import datetime


from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasSGDOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_vectorized import VectorizedDPSGD
from tensorflow.keras.applications.resnet import ResNet152, ResNet50, ResNet101
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# tf.config.experimental.set_synchronous_execution(True)

from squeezenet import SqueezeNet

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 250, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_boolean(
    'logdevice', False, 'If True, set log device placement')
flags.DEFINE_boolean(
    'vectorized', False, 'If True, use vectorized version')
flags.DEFINE_integer('input_size', 32, 'Input size')
flags.DEFINE_string('model', None, 'Model name')
flags.DEFINE_integer('n_batches', 30, 'Number of mini-batch')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

tf.debugging.set_log_device_placement(True)
physical_devices = tf.config.list_physical_devices('TPU')
tf.config.set_visible_devices(physical_devices[0], 'TPU')

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
strategy = tf.distribute.OneDeviceStrategy(device="/TPU:0")
# strategy = tf.distribute.TPUStrategy(resolver)

# policy = mixed_precision.Policy('mixed_bfloat16')
# mixed_precision.set_policy(policy)

def compute_epsilon(steps):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / 60000
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to 1e-5 because MNIST has 60000 training points.
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def load_data():
  """Loads dataset and preprocesses to combine training and validation data."""
  if FLAGS.input_size < 40:
    train_data = np.random.rand(FLAGS.batch_size*FLAGS.n_batches, FLAGS.input_size, FLAGS.input_size, 3)
    train_labels = np.zeros((FLAGS.batch_size*FLAGS.n_batches, 10))
    train_labels[int(FLAGS.batch_size*FLAGS.n_batches/2):,2] = 1
    train_labels[:int(FLAGS.batch_size*FLAGS.n_batches/2),1] = 1

  if FLAGS.input_size > 40:
    train_data = np.random.rand(FLAGS.batch_size*FLAGS.n_batches, FLAGS.input_size, FLAGS.input_size, 3)
    train_labels = np.zeros((FLAGS.batch_size*FLAGS.n_batches, 10))
    train_labels[int(FLAGS.batch_size*FLAGS.n_batches/2):,2] = 1
    train_labels[:int(FLAGS.batch_size*FLAGS.n_batches/2),1] = 1
  # train_data = np.array(train_data, dtype=np.float32) / 255
  # test_data = np.array(test_data, dtype=np.float32) / 255

  # train_labels = np.array(train_labels, dtype=np.int32)
  # test_labels = np.array(test_labels, dtype=np.int32)

  assert train_data.min() >= 0.
  assert train_data.max() <= 1.
  # assert train_labels.ndim == 2

  return train_data, train_labels


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  # if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
  #   raise ValueError('Number of microbatches should divide evenly batch_size')

  # Load training and test data.
  train_data, train_labels = load_data()

  with strategy.scope():
    if FLAGS.model == "vgg16":
      model = VGG16(include_top=True, weights=None, input_shape=(FLAGS.input_size,FLAGS.input_size,3), classes=10)
    if FLAGS.model == "resnet50":
      model = ResNet50(include_top=True, weights=None, input_shape=(FLAGS.input_size,FLAGS.input_size,3), classes=10)
    if FLAGS.model == "resnet152":
      model = ResNet152(include_top=True, weights=None, input_shape=(FLAGS.input_size,FLAGS.input_size,3), classes=10)
    if FLAGS.model == "squeezenetv1.1":
      model = SqueezeNet(include_top=True, weights=None, input_shape=(FLAGS.input_size,FLAGS.input_size,3), classes=10)
    if FLAGS.model == "mobilenetv3small":
      model = MobileNetV3Small(include_top=True, weights=None, input_shape=(FLAGS.input_size,FLAGS.input_size,3), classes=10)
    if FLAGS.model == "debug":
      model = tf.keras.Sequential()
      model.add(tf.keras.layers.AveragePooling2D(pool_size=(16, 16), strides=16, padding="valid", name="AvgPool"))
      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.Dense(10))

    print(model)

    if FLAGS.dpsgd:
      if not FLAGS.vectorized:
        print("Unvectorized DP-SGD will be used.")
        optimizer = DPKerasSGDOptimizer(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate)
      else:
        print("Vectorized DP-SGD will be used.")
        optimizer = VectorizedDPKerasSGDOptimizer(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate)
      # Compute vector of per-example loss rather than its mean over a minibatch.
      loss = tf.keras.losses.CategoricalCrossentropy(
          from_logits=False, reduction=tf.losses.Reduction.NONE)
    else:
      print("Vanilla SGD will be used.")
      optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
      loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    # # Compile model with Keras
    # model.compile(optimizer=optimizer, loss=loss, steps_per_execution=1, metrics=['accuracy'], run_eagerly=False)

  # Create a TensorBoard callback
  if not FLAGS.dpsgd:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.losses.Reduction.SUM)
  logs = "logs/" + FLAGS.model + "_" + str(FLAGS.input_size) + "_" + str(FLAGS.batch_size) + "_" + ("DPSGD" if FLAGS.dpsgd else "SGD") + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

  tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                  histogram_freq = 0,
                                                  profile_batch = 1)

  train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(FLAGS.batch_size)

  def get_per_sample_gradients(arg):
    with tf.keras.backend.name_scope("computing_grads"):
      with tf.GradientTape() as g:
        inp, label = arg
        inp = tf.expand_dims(inp, 0)
        label = tf.expand_dims(label, 0)
        prediction = model(inp)
        loss_value = loss(label, prediction)


      per_sample_grads = g.gradient(loss_value, (model.trainable_weights))

      def clip_gradients_vmap(g, l2_norm_clip):
        """Clips gradients in a way that is compatible with vectorized_map."""
        grads_flat = tf.nest.flatten(g)
        squared_l2_norms = [
            tf.reduce_sum(input_tensor=tf.square(g)) for g in grads_flat
        ]
        global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
        div = tf.maximum(global_norm / l2_norm_clip, 1.)
        clipped_flat = [g / div for g in grads_flat]
        clipped_grads = tf.nest.pack_sequence_as(g, clipped_flat)
        return clipped_grads

      clipped_grads = clip_gradients_vmap(per_sample_grads, 2.0)
    
    return clipped_grads

  def clip_gradients(per_sample_grads):
    # grads_flat = tf.nest.flatten(per_sample_grads)
    with tf.keras.backend.name_scope("computing_norms"):
      squared_l2_norms = [
          tf.reshape(tf.square(tf.norm(g)), [1]) for g in per_sample_grads
      ]
      global_norm = tf.norm(tf.concat(squared_l2_norms, axis=0))

    with tf.keras.backend.name_scope("clipping_grads"):
      div = tf.maximum(global_norm / 2.0, 1.)
      clipped_flat = [g / div for g in per_sample_grads]
      clipped_grads = tf.nest.pack_sequence_as(per_sample_grads, clipped_flat)

    return clipped_grads

  @tf.function
  def reduce_noise_normalize_batch(stacked_grads):
    with tf.keras.backend.name_scope("add_noise_and_reduce"):
      summed_grads = tf.reduce_sum(input_tensor=stacked_grads, axis=0)
      noise_stddev = 2.0 * 0.5
      noise = tf.random.normal(
          tf.shape(input=summed_grads), stddev=noise_stddev)
      noised_grads = summed_grads + noise
      return noised_grads / tf.cast(FLAGS.batch_size, tf.float32)

      # Normalize by number of microbatches and return.

    # return final_gradients 

  @tf.function
  def get_per_batch_gradients(arg):
    with tf.keras.backend.name_scope("computing_grads"):
      inp, label = arg
      with tf.GradientTape(persistent=False) as g:
        
        # inp, label = arg
        # inp = tf.expand_dims(inp)
        # label = tf.expand_dims(label)
        prediction = model(inp)
        loss_value_raw = loss(label, prediction)
        loss_value = tf.reduce_sum(loss_value_raw)


      per_batch_grads = g.gradient(loss_value, (model.trainable_weights))
    
    return per_batch_grads

  # @tf.function(jit_compile=True)
  # @tf.function
  def get_private_grads(x_batch_train, y_batch_train, step):
    
    if FLAGS.dpsgd:
      clipped_gradients = tf.vectorized_map(get_per_sample_gradients, (x_batch_train, y_batch_train))

      # tf.profiler.experimental.start(logs)
      # clipped_gradients = tf.vectorized_map(clip_gradients, per_sample_grads)

      final_gradients = tf.nest.map_structure(reduce_noise_normalize_batch,
                                                  clipped_gradients)

    else:
      per_sample_grads = None
      # tf.profiler.experimental.start(logs)
      final_gradients = get_per_batch_gradients((x_batch_train, y_batch_train))

    with tf.keras.backend.name_scope("update_params"):
      optimizer._was_dp_gradients_called = True
      optimizer.apply_gradients(zip(final_gradients, model.trainable_weights))

    # tf.profiler.experimental.stop()
    # if FLAGS.dpsgd:
    #   tf.print(per_sample_grads)
    # tf.print(final_gradients)

    return final_gradients
  
    # return clipped_gradients

  # tf.profiler.experimental.start(logs)

  # with tf.device('/TPU:0'):
  per_sample_grads_list = []
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

    if step == 4:
      tf.profiler.experimental.start(logs, tf.profiler.experimental.ProfilerOptions(delay_ms=0))
    
    # per_example_gradients = get_private_grads(x_batch_train, y_batch_train)
    final_gradients = strategy.run(get_private_grads, (x_batch_train, y_batch_train, step))
    
    print(final_gradients)
    # strategy.run(clip_gradients, ((x_batch_train, y_batch_train),))

    # optimizer._was_dp_gradients_called = True
    # optimizer.apply_gradients(zip(per_example_gradients, model.trainable_weights))

    if step == 4:
      tf.profiler.experimental.stop()
  
  # tf.profiler.experimental.stop()

  # # Train model with Keras
  # model.fit(train_data, train_labels,
  #           epochs=FLAGS.epochs,
  #           batch_size=FLAGS.batch_size,
  #           callbacks = [tboard_callback])

  # Compute the privacy budget expended.
  if FLAGS.dpsgd:
    eps = compute_epsilon(FLAGS.epochs * 60000 // FLAGS.batch_size)
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)
  else:
    print('Trained with vanilla non-private SGD optimizer')

  with open(logs+"success", "w") as f:
    f.write("success")

  for root, subdirs, files in os.walk("logs"):
    # print(root, subdirs, files)
    for file in files:
      if "xplane.pb" in file:
        os.remove(os.path.join(root, file))

if __name__ == '__main__':
  app.run(main)
