import os
import pathlib
from re import I

models = ["lstm-large"]
seq_lengths = ["256"] #"32", "64", "128", "256"
algos = ["Reweight"] #"SGD", "DPSGD", 

for model in models:
    for seq_length in seq_lengths:
        for algo in algos:
            # batch_size = start_batch_size[(model, input_size, algo)]
            batch_size = 256
            
            successes= [0]
            fails = []

            while True:
                success = False
                for root, subdirs, files in os.walk("logs"):
                    for file in files:
                        if f"{model}_{seq_length}_{batch_size}_{algo}" in file:
                            success = True

                if not success:
                    if algo == "Reweight":
                        print(f"python3 tutorials_new/dpsgd_reweight_lstm_optimized.py --epochs 1 --model {model} --seq_length {seq_length} --batch_size {batch_size} --dpsgd=True --vectorized --n_batches 3")
                        os.system(f"python3 tutorials_new/dpsgd_reweight_lstm_optimized.py --epochs 1 --model {model} --seq_length {seq_length} --batch_size {batch_size} --dpsgd=True --vectorized --n_batches 3 > temp")
                    elif algo == "DPSGD":
                        print(f"python3 tutorials_new/dpsgd_lstm_perf.py --epochs 1 --model {model} --seq_length {seq_length} --batch_size {batch_size} --dpsgd=True --vectorized --n_batches 3")
                        os.system(f"python3 tutorials_new/dpsgd_lstm_perf.py --epochs 1 --model {model} --seq_length {seq_length} --batch_size {batch_size} --dpsgd=True --vectorized --n_batches 3 > temp")
                    else:
                        print(f"python3 tutorials_new/dpsgd_lstm_perf.py --epochs 1 --model {model} --seq_length {seq_length} --batch_size {batch_size} --dpsgd=False --n_batches 3")
                        os.system(f"python3 tutorials_new/dpsgd_lstm_perf.py --epochs 1 --model {model} --seq_length {seq_length} --batch_size {batch_size} --dpsgd=False --n_batches 3 > temp")


                success = False
                for root, subdirs, files in os.walk("logs"):
                    for file in files:
                        if f"{model}_{seq_length}_{batch_size}_{algo}" in file:
                            success = True

                if success:
                    batch_size = batch_size * 2
                    continue
                else:
                    break