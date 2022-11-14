import jax
from jax.lib import xla_client
from jax import jit
import time
import numpy as np
from jax import numpy as jnp
import statistics
import sys
import traceback

n_repeat = 1000 # measure ms

forward_on = True
data_backprop_on = True
per_batch_backprop_on = True
per_sample_backprop_on = True

key = jax.random.PRNGKey(0)

def gemm_func(A, B):
  return (jnp.matmul(A, B, precision=jax.lax.Precision.DEFAULT),)

def gemm_and_norm_func(A, B):
  C = jnp.matmul(A, B, precision=jax.lax.Precision.DEFAULT)
  C_norm = jnp.linalg.norm(C)
  return C, C_norm

def norm_func(A, B):
  C = jnp.matmul(A, B, precision=jax.lax.Precision.DEFAULT)
  C_norm = jnp.linalg.norm(C)
  return (C_norm,)

gemm_jit = jit(gemm_func)
gemm_and_norm_jit = jit(gemm_and_norm_func)
norm_jit = jit(norm_func)

def profile(func, A_shape, B_shape, n_trials=5, warm_up=2):
  A = jax.random.normal(key, A_shape, dtype=jax.numpy.bfloat16)
  B = jax.random.normal(key, B_shape, dtype=jax.numpy.bfloat16)
  jax.device_put(A, jax.devices()[0])
  jax.device_put(B, jax.devices()[0])

  A.block_until_ready()
  A.block_host_until_ready()
  B.block_until_ready()
  B.block_host_until_ready()

  excepted = False
  try:
    for i in range(warm_up):
      C = func(A, B)
      for c in C:
        c.block_until_ready()
  except:
    traceback.print_exc()
    excepted = True

  try:
    trials_time_list = []
    for i in range(n_trials):
      start = time.time()
      if excepted:
        for i in range(n_repeat):
          C = func(A, B)
      else:
        for i in range(n_repeat):
          C = func(A, B)

      for c in C:
        c.block_until_ready()
        c.block_host_until_ready()

      end = time.time()

      trials_time_list += [(end - start)]

    print(jax.numpy.shape(C))

  except:
    trials_time_list = [1, 1, 1]


  return sum(trials_time_list[2:])/len(trials_time_list[2:])

# run_one_operand(only_norm, "exp_logs/norm_128_128", (128, 128))
# run_one_operand(only_norm, "exp_logs/norm_128_4096", (128, 4096))
# run_one_operand(only_norm, "exp_logs/norm_4096_4096", (4096, 4096))

def run(m, k, n):
  print("==========================================")
  print(f"{m}, {k}, {n}")
  print(profile(gemm_jit, (m, k), (k, n)))
  print(profile(gemm_and_norm_jit, (m, k), (k, n)))
  print(profile(norm_jit, (m, k), (k, n)))
  

run(128, 128, 128)
run(128, 4096, 128)
run(4096, 128, 128)
run(128, 128, 4096)
run(4096, 128, 4096)