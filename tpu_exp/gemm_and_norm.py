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

@tf.function
def gemm(A, B):
  C = tf.matmul(A, B)
  return C

@tf.function
def gemm_and_norm(A, B):
  C = tf.matmul(A, B)
  C_norm = tf.norm(C, axis=(-2, -1))
  return C, C_norm

@tf.function
def norm(A, B):
  C = tf.matmul(A, B)
  C_norm = tf.norm(C, axis=(-2, -1))
  return C_norm

@tf.function
def only_norm(C):
  C_norm = tf.norm(C, axis=(-2, -1))
  return C_norm

@tf.function
def read_and_write(C):
  C = C[0, 0] + 1
  return C

def run(func, logs, A_shape, B_shape, batch_size=1):
  try:
    with tf.device('/TPU:0'):
      A = tf.random.normal((batch_size, *A_shape), dtype=tf.bfloat16)
      B = tf.random.normal((batch_size, *B_shape), dtype=tf.bfloat16)
      for step in range(100):
        if step == 10:
          tf.profiler.experimental.start(logs, tf.profiler.experimental.ProfilerOptions(delay_ms=0))

        result = func(A, B)
        print(result)

        if step == 95:
          tf.profiler.experimental.stop()
  except:
    pass

def run_one_operand(func, logs, A_shape):

  with tf.device('/TPU:0'):
    A = tf.random.normal((A_shape), dtype=tf.bfloat16)
    for step in range(100):
      if step == 10:
        tf.profiler.experimental.start(logs, tf.profiler.experimental.ProfilerOptions(delay_ms=0))

      result = func(A)
      if step == 95:
        print(result)

      if step == 95:
        tf.profiler.experimental.stop()

# for b in [1, 4, 16, 64, 256, 1024]:
#   for i in [128, 256, 512, 1024, 2048]:
#     run_one_operand(only_norm, f"norm_logs/only_norm_{b}_{i}_{i}", (b, i, i))
#     for root, subdirs, files in os.walk("norm_logs"):
#       # print(root, subdirs, files)
#       for file in files:
#         if "trace.json.gz" not in file and "events.out" not in file:
#           os.remove(os.path.join(root, file))

# for i in range(1024, 1024*1024*1024, 1024*1024):
#   run_one_operand(only_norm, f"norm_logs/only_norm_{i}", (i, ))
#   for root, subdirs, files in os.walk("norm_logs"):
#     # print(root, subdirs, files)
#     for file in files:
#       if "trace.json.gz" not in file and "events.out" not in file:
#         os.remove(os.path.join(root, file))

# run(gemm, "exp_logs/gemm_1_1024_16_1024", (1024, 16), (16, 1024))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_1_1024_16_1024", (1024, 16), (16, 1024))
run(norm, "exp_logs/norm_1_1024_16_1024", (1024, 16), (16, 1024))

# m=128
# k=128
# for b in [1, 4, 16]:
#   for n in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]:
#     run(gemm, f"exp_logs/gemm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(gemm_and_norm, f"exp_logs/gemm_and_norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(norm, f"exp_logs/norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)

# m=128
# n=128
# for b in [1, 4, 16]:
#   for k in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]:
#     run(gemm, f"exp_logs/gemm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(gemm_and_norm, f"exp_logs/gemm_and_norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(norm, f"exp_logs/norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)

# m=128
# n=128
# for b in [1, 4, 16, 64, 256, 1024]:
#   for k in [2, 512]:
#     run(gemm, f"exp_logs/gemm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(gemm_and_norm, f"exp_logs/gemm_and_norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(norm, f"exp_logs/norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)

# m=256
# n=256
# for b in [1, 4, 16, 64, 256, 1024]:
#   for k in [2, 512]:
#     run(gemm, f"exp_logs/gemm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(gemm_and_norm, f"exp_logs/gemm_and_norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(norm, f"exp_logs/norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)


# m=512
# n=512
# for b in [1, 4, 16, 64, 256, 1024]:
#   for k in [2, 512]:
#     run(gemm, f"exp_logs/gemm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(gemm_and_norm, f"exp_logs/gemm_and_norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(norm, f"exp_logs/norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)

# m=1024
# n=1024
# for b in [1, 4, 16, 64, 256, 1024]:
#   for k in [2, 512]:
#     run(gemm, f"exp_logs/gemm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(gemm_and_norm, f"exp_logs/gemm_and_norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(norm, f"exp_logs/norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)

# m=2048
# n=2048
# for b in [1, 4, 16, 64, 256, 1024]:
#   for k in [4, 8, 16, 32, 64, 128, 256]:
#     run(gemm, f"exp_logs/gemm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(gemm_and_norm, f"exp_logs/gemm_and_norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(norm, f"exp_logs/norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)

# m=4096
# n=4096
# for b in [1, 4, 16, 64, 256, 1024]:
#   for k in [4, 8, 16, 32, 64, 128, 256]:
#     run(gemm, f"exp_logs/gemm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(gemm_and_norm, f"exp_logs/gemm_and_norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)
#     run(norm, f"exp_logs/norm_{b}_{m}_{k}_{n}", (m, k), (k, n), batch_size=b)

# run(gemm, "exp_logs/gemm_256_128_128", (256, 128), (128, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_256_128_128", (256, 128), (128, 128))
# run(norm, "exp_logs/norm_256_128_128", (256, 128), (128, 128))

# run(gemm, "exp_logs/gemm_512_128_128", (512, 128), (128, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_512_128_128", (512, 128), (128, 128))
# run(norm, "exp_logs/norm_512_128_128", (512, 128), (128, 128))

# run(gemm, "exp_logs/gemm_1024_128_128", (1024, 128), (128, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_1024_128_128", (1024, 128), (128, 128))
# run(norm, "exp_logs/norm_1024_128_128", (1024, 128), (128, 128))

# run(gemm, "exp_logs/gemm_2048_128_128", (2048, 128), (128, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_2048_128_128", (2048, 128), (128, 128))
# run(norm, "exp_logs/norm_2048_128_128", (2048, 128), (128, 128))

# run(gemm, "exp_logs/gemm_4096_128_128", (4096, 128), (128, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_4096_128_128", (4096, 128), (128, 128))
# run(norm, "exp_logs/norm_4096_128_128", (4096, 128), (128, 128))

# run(gemm, "exp_logs/gemm_8192_128_128", (8192, 128), (128, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_8192_128_128", (8192, 128), (128, 128))
# run(norm, "exp_logs/norm_8192_128_128", (8192, 128), (128, 128))

# run(gemm, "exp_logs/gemm_16384_128_128", (16384, 128), (128, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_16384_128_128", (16384, 128), (128, 128))
# run(norm, "exp_logs/norm_16384_128_128", (16384, 128), (128, 128))

# run(gemm, "exp_logs/gemm_32768_128_128", (32768, 128), (128, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_32768_128_128", (32768, 128), (128, 128))
# run(norm, "exp_logs/norm_32768_128_128", (32768, 128), (128, 128))

# run(gemm, "exp_logs/gemm_65536_128_128", (65536, 128), (128, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_65536_128_128", (65536, 128), (128, 128))
# run(norm, "exp_logs/norm_65536_128_128", (65536, 128), (128, 128))

# run(gemm, "exp_logs/gemm_131072_128_128", (131072, 128), (128, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_131072_128_128", (131072, 128), (128, 128))
# run(norm, "exp_logs/norm_131072_128_128", (131072, 128), (128, 128))

# run(gemm, "exp_logs/gemm_262144_128_128", (262144, 128), (128, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_262144_128_128", (262144, 128), (128, 128))
# run(norm, "exp_logs/norm_262144_128_128", (262144, 128), (128, 128))

# run(gemm, "exp_logs/gemm_524288_128_128", (524288, 128), (128, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_524288_128_128", (524288, 128), (128, 128))
# run(norm, "exp_logs/norm_524288_128_128", (524288, 128), (128, 128))

# run(gemm, "exp_logs/gemm_128_4096_128", (128, 4096), (4096, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_128_4096_128", (128, 4096), (4096, 128))
# run(norm, "exp_logs/norm_128_4096_128", (128, 4096), (4096, 128))

# run(gemm, "exp_logs/gemm_4096_128_128", (4096, 128), (128, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_4096_128_128", (4096, 128), (128, 128))
# run(norm, "exp_logs/norm_4096_128_128", (4096, 128), (128, 128))

# run(gemm, "exp_logs/gemm_128_128_4096", (128, 128), (128, 4096))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_128_128_4096", (128, 128), (128, 4096))
# run(norm, "exp_logs/norm_128_128_4096", (128, 128), (128, 4096))

# run(gemm, "exp_logs/gemm_4096_128_4096", (4096, 128), (128, 4096))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_4096_128_4096", (4096, 128), (128, 4096))
# run(norm, "exp_logs/norm_4096_128_4096", (4096, 128), (128, 4096))

# run(gemm, "exp_logs/gemm_128_8192_128", (128, 8192), (8192, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_128_8192_128", (128, 8192), (8192, 128))
# run(norm, "exp_logs/norm_128_8192_128", (128, 8192), (8192, 128))

# run(gemm, "exp_logs/gemm_128_16384_128", (128, 16384), (16384, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_128_16384_128", (128, 16384), (16384, 128))
# run(norm, "exp_logs/norm_128_16384_128", (128, 16384), (16384, 128))

# run(gemm, "exp_logs/gemm_128_32768_128", (128, 32768), (32768, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_128_32768_128", (128, 32768), (32768, 128))
# run(norm, "exp_logs/norm_128_32768_128", (128, 32768), (32768, 128))

# run_one_operand(only_norm, "exp_logs/norm_64_64", (64, 64))

# run_one_operand(only_norm, "exp_logs/norm_test", (94692352, 42))

# run(gemm, "exp_logs/gemm_64_64_64", (64, 64), (64, 64))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_64_64_64", (64, 64), (64, 64))
# run(norm, "exp_logs/norm_64_64_64", (64, 64), (64, 64))

# run(gemm, "exp_logs/gemm_64_64_128", (64, 64), (64, 128))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_64_64_128", (64, 64), (64, 128))
# run(norm, "exp_logs/norm_64_64_128", (64, 64), (64, 128))

# run(gemm, "exp_logs/gemm_64_64_256", (64, 64), (64, 256))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_64_64_256", (64, 64), (64, 256))
# run(norm, "exp_logs/norm_64_64_256", (64, 64), (64, 256))

# run(gemm, "exp_logs/gemm_64_64_512", (64, 64), (64, 512))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_64_64_512", (64, 64), (64, 512))
# run(norm, "exp_logs/norm_64_64_512", (64, 64), (64, 512))

# run(gemm, "exp_logs/gemm_64_64_1024", (64, 64), (64, 1024))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_64_64_1024", (64, 64), (64, 1024))
# run(norm, "exp_logs/norm_64_64_1024", (64, 64), (64, 1024))

# run(gemm, "exp_logs/gemm_64_64_2048", (64, 64), (64, 2048))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_64_64_2048", (64, 64), (64, 2048))
# run(norm, "exp_logs/norm_64_64_2048", (64, 64), (64, 2048))

# run(gemm, "exp_logs/gemm_64_256_64", (64, 256), (256, 64))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_64_256_64", (64, 256), (256, 64))
# run(norm, "exp_logs/norm_64_256_64", (64, 256), (256, 64))

# run(gemm, "exp_logs/gemm_64_512_64", (64, 512), (512, 64))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_64_512_64", (64, 512), (512, 64))
# run(norm, "exp_logs/norm_64_512_64", (64, 512), (512, 64))

# run(gemm, "exp_logs/gemm_64_1024_64", (64, 1024), (1024, 64))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_64_1024_64", (64, 1024), (1024, 64))
# run(norm, "exp_logs/norm_64_1024_64", (64, 1024), (1024, 64))

# run(gemm, "exp_logs/gemm_64_2048_64", (64, 2048), (2048, 64))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_64_2048_64", (64, 2048), (2048, 64))
# run(norm, "exp_logs/norm_64_2048_64", (64, 2048), (2048, 64))

# run(gemm, "exp_logs/gemm_4096_16_4096", (4096, 16), (16, 4096))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_4096_16_4096", (4096, 16), (16, 4096))
# run(norm, "exp_logs/norm_4096_16_4096", (4096, 16), (16, 4096))

# run(gemm, "exp_logs/gemm_4096_32_4096", (4096, 32), (32, 4096))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_4096_32_4096", (4096, 32), (32, 4096))
# run(norm, "exp_logs/norm_4096_32_4096", (4096, 32), (32, 4096))

# run(gemm, "exp_logs/gemm_4096_64_4096", (4096, 64), (64, 4096))
# run(gemm_and_norm, "exp_logs/gemm_and_norm_4096_64_4096", (4096, 64), (64, 4096))
# run(norm, "exp_logs/norm_4096_64_4096", (4096, 64), (64, 4096))

for root, subdirs, files in os.walk("exp_logs"):
  # print(root, subdirs, files)
  for file in files:
    if "trace.json.gz" not in file and "events.out" not in file:
      os.remove(os.path.join(root, file))