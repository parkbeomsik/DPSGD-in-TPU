import jax
import time
import numpy as np
import statistics

f_name = "layer_configs/220228_cnn_resnet50_imagenet_mm.csv"

per_batch_backprop_gemm = []
per_sample_backprop_gemm = []
with open(f_name, "r", encoding='utf-8-sig') as f:
  while True:
    line = f.readline()
    if not line:
        break
    gemm = (1, int(line.split(',')[0]), int(line.split(',')[1]), int(line.split(',')[2]))
    per_batch_backprop_gemm += [gemm]

    gemm = (int(line.split(',')[4]), int(line.split(',')[5]), int(line.split(',')[6]), int(line.split(',')[7]))
    per_sample_backprop_gemm += [gemm]

n_trials = 50

# jax.profiler.start_trace("./tensorboard")

per_sample_result_dict = {}
# with jax.profiler.StepTraceAnnotation("train", step_num=0):
for gemm in per_sample_backprop_gemm:
    print(f"Current gemm for b={gemm[0]},m={gemm[1]},k={gemm[2]},n={gemm[3]}")
    key = jax.random.PRNGKey(0)
    b = gemm[0]
    m = gemm[1]
    k = gemm[2]
    n = gemm[3]

    # with jax.profiler.TraceAnnotation(f"b={b},m={m},k={k},n={n}"):
    trial_list = []
    for i in range(n_trials):
      A = jax.random.normal(key, (b, m, k), dtype=np.float32)
      B = jax.random.normal(key, (b, n, k), dtype=np.float32)
      B = jax.numpy.transpose(B, (0, 2, 1))
      jax.device_put(A, jax.devices()[0])
      jax.device_put(B, jax.devices()[0])

      A.block_until_ready()
      B.block_until_ready()
      start = time.time()

      # Run the operations to be profiled
      C = jax.lax.batch_matmul(A, B, precision=jax.lax.Precision.DEFAULT)
      C.block_until_ready()
      end = time.time()

      trial_list += [end - start]

    per_sample_result_dict[gemm] = trial_list

per_batch_result_dict = {}
# with jax.profiler.StepTraceAnnotation("train", step_num=0):
for gemm in per_batch_backprop_gemm:
    print(f"Current gemm for b={gemm[0]},m={gemm[1]},k={gemm[2]},n={gemm[3]}")
    key = jax.random.PRNGKey(0)
    b = gemm[0]
    m = gemm[1]
    k = gemm[2]
    n = gemm[3]

    # with jax.profiler.TraceAnnotation(f"b={b},m={m},k={k},n={n}"):
    trial_list = []
    for i in range(n_trials):
      A = jax.random.normal(key, (1, m, k), dtype=np.float32)
      B = jax.random.normal(key, (1, n, k), dtype=np.float32)
      B = jax.numpy.transpose(B, (0, 2, 1))
      jax.device_put(A, jax.devices()[0])
      jax.device_put(B, jax.devices()[0])

      A.block_until_ready()
      B.block_until_ready()
      start = time.time()

      # Run the operations to be profiled
      C = jax.numpy.matmul(A, B, precision=jax.lax.Precision.DEFAULT)
      C.block_until_ready()
      end = time.time()

      trial_list += [end - start]

    per_batch_result_dict[gemm] = trial_list

with open(f_name.split(".")[-2] + "_tpu_result.csv", "w") as f:
  print("b,m,k,n,avg_time(ms),stdev,,b,m,k,n,avg_time(ms),stdev")
  f.write("b,m,k,n,avg_time(ms),stdev,,b,m,k,n,avg_time(ms),stdev\n")

  for i in range(len(per_sample_backprop_gemm)):
    per_sample_gemm = per_sample_backprop_gemm[i]
    per_batch_gemm = per_batch_backprop_gemm[i]

    per_sample_results = per_sample_result_dict[per_sample_gemm][10:]
    per_sample_log = f"{per_sample_gemm[0]},{per_sample_gemm[1]},{per_sample_gemm[2]},{per_sample_gemm[3]},{sum(per_sample_results)/len(per_sample_results)*1000},{statistics.stdev(per_sample_results)*1000}"

    per_batch_results = per_batch_result_dict[per_batch_gemm][10:]
    per_batch_log = f"{per_batch_gemm[0]},{per_batch_gemm[1]},{per_batch_gemm[2]},{per_batch_gemm[3]},{sum(per_batch_results)/len(per_batch_results)*1000},{statistics.stdev(per_batch_results)*1000}"

    print(per_batch_log + ",," + per_sample_log)
    f.write(per_batch_log + ",," + per_sample_log + '\n')

# jax.profiler.stop_trace()
