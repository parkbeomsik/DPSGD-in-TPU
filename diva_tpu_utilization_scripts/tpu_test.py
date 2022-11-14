import jax
from jax.lib import xla_client
from jax import jit
import time
import numpy as np
import statistics
import sys

n_repeat = 1000 # measure ms

forward_on = True
data_backprop_on = True
per_batch_backprop_on = True
per_sample_backprop_on = True

key = jax.random.PRNGKey(0)

def batched_gemm(A, B):
  return jax.lax.batch_matmul(A, B, precision=jax.lax.Precision.DEFAULT)

def profile_batched_gemm(A_shape, B_shape, n_trials, warm_up):
  A = jax.random.normal(key, A_shape, dtype=jax.numpy.bfloat16)
  B = jax.random.normal(key, B_shape, dtype=jax.numpy.bfloat16)
  jax.device_put(A, jax.devices()[0])
  jax.device_put(B, jax.devices()[0])

  A.block_until_ready()
  A.block_host_until_ready()
  B.block_until_ready()
  B.block_host_until_ready()

  batched_gemm_jit = jit(batched_gemm)

  excepted = False
  try:
    for i in range(warm_up):
      batched_gemm_jit(A, B).block_until_ready()
  except:
    traceback.print_exc()
    excepted = True

  trials_time_list = []
  for i in range(n_trials):
    start = time.time()
    if excepted:
      for i in range(n_repeat):
        C = batched_gemm(A, B)
    else:
      for i in range(n_repeat):
        C = batched_gemm_jit(A, B)
    C.block_until_ready()
    C.block_host_until_ready()
    end = time.time()

    trials_time_list += [(end - start)]

  print(jax.numpy.shape(C))

  return trials_time_list

A = jax.random.normal(key, (1024, 1024, 1024), dtype=jax.numpy.bfloat16)