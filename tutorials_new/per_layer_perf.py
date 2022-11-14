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
from tensorflow.keras.mixed_precision import experimental as mixed_precision

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 4, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 4, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_boolean(
    'logdevice', False, 'If True, set log device placement')
flags.DEFINE_boolean(
    'vectorized', False, 'If True, use vectorized version')
flags.DEFINE_integer('seq_length', 128, 'Sequence length')
flags.DEFINE_integer('hidden_size', 768, 'Hidden size')
flags.DEFINE_integer('embedding_size', 512, 'Embedding size')
flags.DEFINE_integer('n_lstm_layers', 1, 'N lstm layers')
flags.DEFINE_string('model', None, 'Model name')
flags.DEFINE_integer('n_batches', 30, 'Number of mini-batch')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

tf.debugging.set_log_device_placement(False)
physical_devices = tf.config.list_physical_devices('TPU')
tf.config.set_visible_devices(physical_devices[0], 'TPU')

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
strategy = tf.distribute.OneDeviceStrategy(device="/TPU:0")
# strategy = tf.distribute.TPUStrategy(resolver)
# strategy = tf.distribute.get_strategy()

# policy = mixed_precision.Policy('mixed_float16')
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
  # train_data = np.random.rand(FLAGS.batch_size*FLAGS.n_batches, FLAGS.seq_length, FLAGS.embedding_size)
  train_data = np.random.randint(0, 200, size=(FLAGS.batch_size*FLAGS.n_batches, FLAGS.seq_length))
  train_labels = np.random.randint(0, 2, size=(FLAGS.batch_size*FLAGS.n_batches, 18))

  # train_data = np.array(train_data, dtype=np.float32) / 255
  # test_data = np.array(test_data, dtype=np.float32) / 255

  # train_labels = np.array(train_labels, dtype=np.int32)
  # test_labels = np.array(test_labels, dtype=np.int32)

  return train_data, train_labels


def main(unused_argv):
  logging.set_verbosity(logging.INFO)

  # Load training and test data.
  train_data, train_labels = load_data()

  with strategy.scope():
    if FLAGS.model is not None:
      if FLAGS.model == "lstm-small":
        FLAGS.embedding_size = 128
        FLAGS.hidden_size = 128
        FLAGS.n_lstm_layers = 1
      if FLAGS.model == "lstm-large":
        FLAGS.embedding_size = 512
        FLAGS.hidden_size = 1024
        FLAGS.n_lstm_layers = 5

    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Input(shape=(FLAGS.seq_length,)))
    # model.add(tf.keras.layers.InputLayer(input_tensor=tf.zeros((FLAGS.seq_length, FLAGS.batch_size))))
    model.add(tf.keras.layers.Embedding(input_dim=259, output_dim=FLAGS.embedding_size, input_length=FLAGS.seq_length)) # 
    # model.add(tf.keras.layers.Permute((0, 2, 1), input_shape=(FLAGS.batch_size, FLAGS.seq_length, FLAGS.embedding_size))) # For time_major=True in LSTM layer
    # model.add(tf.keras.layers.Input(batch_shape=(FLAGS.batch_size, FLAGS.seq_length, FLAGS.embedding_size)))
    # model.add(tf.keras.layers.InputLayer(input_shape=(FLAGS.seq_length, FLAGS.embedding_size), batch_size=FLAGS.batch_size))
    model.add(tf.keras.layers.InputLayer(input_tensor=tf.zeros((FLAGS.batch_size, FLAGS.seq_length, FLAGS.embedding_size))))
    for i in range(FLAGS.n_lstm_layers):
      if i == 0:
        batch_input_shape = (FLAGS.batch_size, FLAGS.seq_length, FLAGS.embedding_size)
        
      if i == FLAGS.n_lstm_layers - 1:
        if not (i == 0):
          batch_input_shape = (FLAGS.batch_size, FLAGS.seq_length, FLAGS.hidden_size)
        return_sequences = False
      else:
        return_sequences = True
        
      model.add(tf.keras.layers.LSTM(FLAGS.hidden_size, input_length=FLAGS.seq_length, batch_input_shape=batch_input_shape, return_sequences=return_sequences, unroll=True)) # 
    model.add(tf.keras.layers.Dense(18, batch_input_shape=(FLAGS.batch_size, FLAGS.hidden_size)))

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
        optimizer = VectorizedDPSGD(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate)
      # Compute vector of per-example loss rather than its mean over a minibatch.
      loss = tf.keras.losses.CategoricalCrossentropy(
          from_logits=True, reduction=tf.losses.Reduction.NONE)
    else:
      print("Vanilla SGD will be used.")
      optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
      loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Compile model with Keras
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  # Create a TensorBoard callback
  if not FLAGS.dpsgd:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.losses.Reduction.SUM)
  logs = "logs/" + FLAGS.model + "_" + str(FLAGS.seq_length) + "_" + str(FLAGS.batch_size) + "_" + ("DPSGD" if FLAGS.dpsgd else "SGD") + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

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
        # inp = tf.transpose(inp, (1, 0))
        label = tf.expand_dims(label, 0)
        # label = tf.transpose(label, (1, 0))
        prediction = model(inp)
        loss_value = loss(label, prediction)


      per_sample_grads = g.gradient(loss_value, (model.trainable_weights))
    
    return per_sample_grads

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
  def reduce_noise_normalize_batch(g):
    with tf.keras.backend.name_scope("add_noise_and_reduce"):
      # Sum gradients over all microbatches.
      summed_gradient = tf.reduce_sum(g, axis=0)

      # Add noise to summed gradients.
      noise_stddev = 2.0 * 0.5
      noise = tf.random.normal(
          tf.shape(input=summed_gradient), stddev=noise_stddev)
      noised_gradient = tf.add(summed_gradient, noise)

      final_gradients = tf.truediv(noised_gradient, FLAGS.batch_size)  

      # Normalize by number of microbatches and return.
      return final_gradients 

  @tf.function
  def get_per_batch_gradients(arg):
    with tf.keras.backend.name_scope("computing_grads"):
      # inp, label = arg
      with tf.GradientTape(persistent=False) as g:
        
        inp, label = arg
        # inp = tf.transpose(inp, (1, 0))
        # label = tf.transpose(label, (1, 0))
        prediction = model(inp)
        loss_value_raw = loss(label, prediction)
        loss_value = tf.reduce_sum(loss_value_raw)


      per_batch_grads = g.gradient(loss_value, (model.trainable_weights))
    
    return per_batch_grads

  # @tf.function(jit_compile=True)
  # @tf.function
  def get_private_grads(x_batch_train, y_batch_train, step):
    
    if FLAGS.dpsgd:
      per_sample_grads = tf.vectorized_map(get_per_sample_gradients, (x_batch_train, y_batch_train))

      clipped_gradients = tf.vectorized_map(clip_gradients, per_sample_grads)

      final_gradients = tf.nest.map_structure(reduce_noise_normalize_batch,
                                                  clipped_gradients)
    
    else:
      per_sample_grads = None
      final_gradients = get_per_batch_gradients((x_batch_train, y_batch_train))

    optimizer._was_dp_gradients_called = True
    optimizer.apply_gradients(zip(final_gradients, model.trainable_weights))


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

  # Compute the privacy budget expended.
  if FLAGS.dpsgd:
    eps = compute_epsilon(FLAGS.epochs * 6000000 // FLAGS.batch_size)
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
