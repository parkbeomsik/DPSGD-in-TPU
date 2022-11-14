import jax
from jax.lib import xla_client
from jax import jit
import time
import numpy as np
import statistics
import sys
import traceback

n_repeat = 1000 # measure ms

forward_on = True
data_backprop_on = True
per_batch_backprop_on = True
per_sample_backprop_on = True

key = jax.random.PRNGKey(0)

def batched_gemm(A, B):
  return jax.lax.batch_matmul(A, B, precision=jax.lax.Precision.DEFAULT)

def profile_batched_gemm(A_shape, B_shape, n_trials, warm_up):
  A = jax.random.normal(key, (b, m, k), dtype=jax.numpy.bfloat16)
  B = jax.random.normal(key, (b, k, n), dtype=jax.numpy.bfloat16)
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

  try:
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

  except:
    trials_time_list = [1, 1, 1]


  return trials_time_list

def forward_gemm_from_config(layer_config):
  if layer_config['layer'] == "Conv":
    if 'G' not in layer_config:
      gemm = {}
      gemm['b'] = 1
      gemm['m'] = layer_config['K']
      gemm['k'] = layer_config['C']*layer_config['R']*layer_config['S']
      gemm['n'] = layer_config['N']*layer_config['P']*layer_config['Q']
    else:
      gemm = {}
      gemm['b'] = layer_config['G']
      gemm['m'] = int(layer_config['K'] / layer_config['G'])
      gemm['k'] = int(layer_config['C']*layer_config['R']*layer_config['S'] / layer_config['G'])
      gemm['n'] = layer_config['N']*layer_config['P']*layer_config['Q']
  elif layer_config['layer'] == "FC":
    gemm = {}
    gemm['b'] = 1
    gemm['m'] = layer_config['H']
    gemm['k'] = layer_config['C']
    gemm['n'] = layer_config['N'] 

  return gemm

def data_backward_from_config(layer_config):
  if layer_config['layer'] == "Conv":
    if 'G' not in layer_config:
      gemm = {}
      gemm['b'] = 1
      gemm['m'] = layer_config['C']
      gemm['k'] = layer_config['R']*layer_config['S']*layer_config['K']
      gemm['n'] = layer_config['N']*layer_config['H']*layer_config['W']
    else:
      gemm = {}
      gemm['b'] = layer_config['G']
      gemm['m'] = int(layer_config['C'] / layer_config['G'])
      gemm['k'] = int(layer_config['R']*layer_config['S']*layer_config['K'] / layer_config['G'])
      gemm['n'] = layer_config['N']*layer_config['H']*layer_config['W']
  elif layer_config['layer'] == "FC":
    gemm = {}
    gemm['b'] = 1
    gemm['m'] = layer_config['C']
    gemm['k'] = layer_config['H']
    gemm['n'] = layer_config['N'] 

  return gemm

def per_batch_from_config(layer_config):
  if layer_config['layer'] == "Conv":
    if 'G' not in layer_config:
      gemm = {}
      gemm['b'] = 1
      gemm['m'] = layer_config['K']
      gemm['k'] = layer_config['N']*layer_config['P']*layer_config['Q']
      gemm['n'] = layer_config['C']*layer_config['R']*layer_config['S']
    else:
      gemm = {}
      gemm['b'] = layer_config['G']
      gemm['m'] = int(layer_config['K'] / layer_config['G'])
      gemm['k'] = layer_config['N']*layer_config['P']*layer_config['Q']
      gemm['n'] = int(layer_config['C']*layer_config['R']*layer_config['S'] / layer_config['G'])
  elif layer_config['layer'] == "FC":
    gemm = {}
    gemm['b'] = 1
    gemm['m'] = layer_config['H']
    gemm['k'] = layer_config['N']
    gemm['n'] = layer_config['C'] 

  return gemm

def per_sample_from_config(layer_config):
  if layer_config['layer'] == "Conv":
    if 'G' not in layer_config:
      gemm = {}
      gemm['b'] = layer_config['N']
      gemm['m'] = layer_config['K']
      gemm['k'] = layer_config['P']*layer_config['Q']
      gemm['n'] = layer_config['C']*layer_config['R']*layer_config['S']
    else:
      gemm = {}
      gemm['b'] = layer_config['N'] * layer_config['G']
      gemm['m'] = int(layer_config['K'] / layer_config['G'])
      gemm['k'] = layer_config['P']*layer_config['Q']
      gemm['n'] = int(layer_config['C']*layer_config['R']*layer_config['S'] / layer_config['G'])
  elif layer_config['layer'] == "FC":
    gemm = {}
    gemm['b'] = layer_config['N']
    gemm['m'] = layer_config['H']
    gemm['k'] = 1
    gemm['n'] = layer_config['C'] 

  return gemm

f_name = sys.argv[1]
print(f_name + " ...")

per_sample_backprop_gemm = []
per_batch_backprop_gemm = []
forward_gemm = []
data_backprop_gemm = []
skip_header = False
with open(f_name, "r", encoding='utf-8-sig') as f:
  while True:
    line = f.readline()
    if not line:
        break
    if not skip_header:
      skip_header = True
      continue
    layer_config = {}
    layer_config['layer'] = line.split(',')[0]
    layer_config['N'] = int(line.split(',')[1])
    layer_config['C'] = int(line.split(',')[2])
    layer_config['H'] = int(line.split(',')[3])
    if layer_config['layer'] == "Conv":
      layer_config['W'] = int(line.split(',')[4])
      layer_config['R'] = int(line.split(',')[5])
      layer_config['S'] = int(line.split(',')[6])
      layer_config['K'] = int(line.split(',')[7])
      layer_config['P'] = int(line.split(',')[8])
      layer_config['Q'] = int(line.split(',')[9])
      if len(line.split(',')) >= 11:
        layer_config['G'] = int(line.split(',')[10])

    forward_gemm += [forward_gemm_from_config(layer_config)]
    data_backprop_gemm += [data_backward_from_config(layer_config)]
    per_batch_backprop_gemm += [per_batch_from_config(layer_config)]
    per_sample_backprop_gemm += [per_sample_from_config(layer_config)]

n_trials = 3
warm_up_trials = 2

# jax.profiler.start_trace("./tensorboard")

forward_time_list = []
forward_stdev_list = []
for i in range(len(forward_gemm)):
  gemm_size = forward_gemm[i]
  b = gemm_size['b']
  m = gemm_size['m']
  k = gemm_size['k']
  n = gemm_size['n']

  print(f"layer {i} forward...  ({b}, {m}, {k}) X ({b}, {k}, {n})")

  if forward_on:
    trials_time_list = profile_batched_gemm((b, m, k), (b, k, n), n_trials=n_trials, warm_up=warm_up_trials)
  else:
    trials_time_list = [1 for _ in range(100)]

  forward_time_list += [sum(trials_time_list)/len(trials_time_list)]
  forward_stdev_list += [statistics.stdev(trials_time_list)]

data_backprop_time_list = []
data_backprop_stdev_list = []
for i in range(len(data_backprop_gemm)):
  gemm_size = data_backprop_gemm[i]
  b = gemm_size['b']
  m = gemm_size['m']
  k = gemm_size['k']
  n = gemm_size['n']

  print(f"layer {i} data_backprop... ({b}, {m}, {k}) X ({b}, {k}, {n})")

  if data_backprop_on and i > 0:
    trials_time_list = profile_batched_gemm((b, m, k), (b, k, n), n_trials=n_trials, warm_up=warm_up_trials)
  else:
    trials_time_list = [1 for _ in range(100)]

  data_backprop_time_list += [sum(trials_time_list)/len(trials_time_list)]
  data_backprop_stdev_list += [statistics.stdev(trials_time_list)]

per_batch_time_list = []
per_batch_stdev_list = []
for i in range(len(per_batch_backprop_gemm)):
  gemm_size = per_batch_backprop_gemm[i]
  b = gemm_size['b']
  m = gemm_size['m']
  k = gemm_size['k']
  n = gemm_size['n']

  print(f"layer {i} per_batch... ({b}, {m}, {k}) X ({b}, {k}, {n})")

  if per_batch_backprop_on:
    trials_time_list = profile_batched_gemm((b, m, k), (b, k, n), n_trials=n_trials, warm_up=warm_up_trials)
  else:
    trials_time_list = [1 for _ in range(100)]

  per_batch_time_list += [sum(trials_time_list)/len(trials_time_list)]
  per_batch_stdev_list += [statistics.stdev(trials_time_list)]

per_sample_time_list = []
per_sample_stdev_list = []
for i in range(len(per_sample_backprop_gemm)):
  gemm_size = per_sample_backprop_gemm[i]
  b = gemm_size['b']
  m = gemm_size['m']
  k = gemm_size['k']
  n = gemm_size['n']

  print(f"layer {i} per_sample... ({b}, {m}, {k}) X ({b}, {k}, {n})")

  if per_sample_backprop_on:
    trials_time_list = profile_batched_gemm((b, m, k), (b, k, n), n_trials=n_trials, warm_up=warm_up_trials)
  else:
    trials_time_list = [1 for _ in range(100)]

  per_sample_time_list += [sum(trials_time_list)/len(trials_time_list)]
  per_sample_stdev_list += [statistics.stdev(trials_time_list)]

# peak_macs = 128 * 128 * 700 * (10**6) # tpu-v2
peak_macs = 2 * 128 * 128 * 900 * (10**6) # tpu-v3

result_f_name = "out/" + f_name.split("/")[-1].split(".")[0] + "_tpuv3_bf16_result.csv"
with open(result_f_name, "w") as f:
  print(f"b,m,k,n,avg_time(ms),stdev(ms),utilization(%),,b,m,k,n,avg_time(ms),stdev(ms),utilization(%),,b,m,k,n,avg_time(ms),stdev(ms),utilization(%),,b,m,k,n,avg_time(ms),stdev(ms),utilization(%)")
  f.write(f"b,m,k,n,avg_time(ms),stdev(ms),utilization(%),,b,m,k,n,avg_time(ms),stdev(ms),utilization(%),,b,m,k,n,avg_time(ms),stdev(ms),utilization(%),,b,m,k,n,avg_time(ms),stdev(ms),utilization(%)\n")

  for i in range(len(per_sample_time_list)):
    gemm = forward_gemm[i]
    utilization = gemm['b']*gemm['m']*gemm['k']*gemm['n'] / (forward_time_list[i] * peak_macs / 1000) * 100
    forward_log = f"{gemm['b']},{gemm['m']},{gemm['k']},{gemm['n']},{forward_time_list[i]},{forward_stdev_list[i]},{utilization}"

    gemm = data_backprop_gemm[i]
    utilization = gemm['b']*gemm['m']*gemm['k']*gemm['n'] / (data_backprop_time_list[i] * peak_macs / 1000) * 100
    data_backprop_log = f"{gemm['b']},{gemm['m']},{gemm['k']},{gemm['n']},{data_backprop_time_list[i]},{data_backprop_stdev_list[i]},{utilization}"

    gemm = per_batch_backprop_gemm[i]
    utilization = gemm['b']*gemm['m']*gemm['k']*gemm['n'] / (per_batch_time_list[i] * peak_macs / 1000) * 100
    per_batch_backprop_log = f"{gemm['b']},{gemm['m']},{gemm['k']},{gemm['n']},{per_batch_time_list[i]},{per_batch_stdev_list[i]},{utilization}"

    gemm = per_sample_backprop_gemm[i]
    utilization = gemm['b']*gemm['m']*gemm['k']*gemm['n'] / (per_sample_time_list[i] * peak_macs / 1000) * 100
    per_sample_backprop_log = f"{gemm['b']},{gemm['m']},{gemm['k']},{gemm['n']},{per_sample_time_list[i]},{per_sample_stdev_list[i]},{utilization}"

    print(forward_log + ",," + data_backprop_log + ",," + per_batch_backprop_log + ",," + per_sample_backprop_log)
    f.write(forward_log + ",," + data_backprop_log + ",," + per_batch_backprop_log + ",," + per_sample_backprop_log + "\n")