import os
import pathlib
from re import I
 
models = ["vgg16", "resnet50", "resnet152", "squeezenetv1.1", "mobilenetv3small"] #"vgg16", 
input_sizes = ["64", "128", "256", "224"] # , "64", "128", "256"
algos = ["Reweight"]

start_batch_size = {("vgg16", "32", "DPSGD") : 256,
                    ("vgg16", "32", "SGD") : 8192,      
                    ("vgg16", "32", "Reweight") : 8192,
                    ("vgg16", "64", "DPSGD") : 128,
                    ("vgg16", "64", "SGD") : 256,
                    ("vgg16", "64", "Reweight") : 256,
                    ("vgg16", "128", "DPSGD") : 128,
                    ("vgg16", "128", "SGD") : 256,
                    ("vgg16", "128", "Reweight") : 256,
                    ("vgg16", "256", "DPSGD") : 128,
                    ("vgg16", "256", "SGD") : 256,
                    ("vgg16", "256", "Reweight") : 256,
                    ("resnet50", "32", "DPSGD") : 256,
                    ("resnet50", "32", "SGD") : 8192,
                    ("resnet50", "32", "Reweight") : 8192,
                    ("resnet50", "64", "DPSGD") : 128,
                    ("resnet50", "64", "SGD") : 256,
                    ("resnet50", "64", "Reweight") : 256,
                    ("resnet50", "128", "DPSGD") : 128,
                    ("resnet50", "128", "SGD") : 256,
                    ("resnet50", "128", "Reweight") : 256,
                    ("resnet50", "256", "DPSGD") : 128,
                    ("resnet50", "256", "SGD") : 256,
                    ("resnet50", "256", "Reweight") : 256,
                    ("resnet152", "32", "DPSGD") : 64,
                    ("resnet152", "32", "SGD") : 1024,
                    ("resnet152", "32", "Reweight") : 1024,
                    ("resnet152", "64", "DPSGD") : 64,
                    ("resnet152", "64", "SGD") : 128,
                    ("resnet152", "64", "Reweight") : 128,
                    ("resnet152", "128", "DPSGD") : 64,
                    ("resnet152", "128", "SGD") : 128,
                    ("resnet152", "128", "Reweight") : 128,
                    ("resnet152", "256", "DPSGD") : 64,
                    ("resnet152", "256", "SGD") : 128,
                    ("resnet152", "256", "Reweight") : 128,
                    ("squeezenetv1.1", "32", "DPSGD") : 4096,
                    ("squeezenetv1.1", "32", "SGD") : 32768,
                    ("squeezenetv1.1", "32", "Reweight") : 32768,
                    ("squeezenetv1.1", "64", "DPSGD") : 256,
                    ("squeezenetv1.1", "64", "SGD") : 1024,
                    ("squeezenetv1.1", "64", "Reweight") : 1024,
                    ("squeezenetv1.1", "128", "DPSGD") : 256,
                    ("squeezenetv1.1", "128", "SGD") : 1024,
                    ("squeezenetv1.1", "128", "Reweight") : 1024,
                    ("squeezenetv1.1", "256", "DPSGD") : 256,
                    ("squeezenetv1.1", "256", "SGD") : 1024,
                    ("squeezenetv1.1", "256", "Reweight") : 1024,
                    ("mobilenetv3small", "32", "DPSGD") : 4096,
                    ("mobilenetv3small", "32", "SGD") : 32768,
                    ("mobilenetv3small", "32", "Reweight") : 32768,
                    ("mobilenetv3small", "64", "DPSGD") : 1024,
                    ("mobilenetv3small", "64", "SGD") : 2048,
                    ("mobilenetv3small", "64", "Reweight") : 2048,
                    ("mobilenetv3small", "128", "DPSGD") : 1024,
                    ("mobilenetv3small", "128", "SGD") : 2048,
                    ("mobilenetv3small", "128", "Reweight") : 2048,
                    ("mobilenetv3small", "256", "DPSGD") : 1024,
                    ("mobilenetv3small", "256", "SGD") : 2048,
                    ("mobilenetv3small", "256", "Reweight") : 2048
                    }

for model in models:
    for input_size in input_sizes:
        for algo in algos:
            print(model, input_size, algo)
            # batch_size = start_batch_size[(model, input_size, algo)]
            batch_size = 16
            
            successes= [0]
            fails = []

            while True:
                print(model, input_size, algo, batch_size)

                success = False
                for root, subdirs, files in os.walk("logs"):
                    for file in files:
                        if f"{model}_{input_size}_{batch_size}_{algo}" in file:
                            success = True

                if not success:
                    if algo == "Reweight":
                        print(f"python3 tutorials_new/dpsgd_reweight_cnn_optimized.py --epochs 1 --model {model} --input_size {input_size} --batch_size {batch_size} --dpsgd=True --vectorized --n_batches 3")
                        os.system(f"python3 tutorials_new/dpsgd_reweight_cnn_optimized.py --epochs 1 --model {model} --input_size {input_size} --batch_size {batch_size} --dpsgd=True --vectorized --n_batches 3 > temp")
                    elif algo == "DPSGD":
                        print(f"python3 tutorials_new/dpsgd_cnn_perf.py --epochs 1 --model {model} --input_size {input_size} --batch_size {batch_size} --dpsgd=True --vectorized --n_batches 3")
                        os.system(f"python3 tutorials_new/dpsgd_cnn_perf.py --epochs 1 --model {model} --input_size {input_size} --batch_size {batch_size} --dpsgd=True --vectorized --n_batches 3 > temp")
                    else:
                        print(f"python3 tutorials_new/dpsgd_cnn_perf.py --epochs 1 --model {model} --input_size {input_size} --batch_size {batch_size} --dpsgd=False --n_batches 3")
                        os.system(f"python3 tutorials_new/dpsgd_cnn_perf.py --epochs 1 --model {model} --input_size {input_size} --batch_size {batch_size} --dpsgd=False --n_batches 3 > temp")


                success = False
                for root, subdirs, files in os.walk("logs"):
                    for file in files:
                        if f"{model}_{input_size}_{batch_size}_{algo}" in file:
                            success = True

                if success:
                    batch_size = batch_size * 2
                    continue
                else:
                    break