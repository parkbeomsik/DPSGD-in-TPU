import gzip
import json
from turtle import update
import os

result = {}

def parse(path):
    print("----------------------------------------")
    print(path[:path.find("2022")].split("/")[1])
    full_name = path[:path.find("2022")].split("/")[1]

    if "gemm_and_norm" in full_name:
        exp = "gemm_and_norm"
        mkn = full_name.split("_", 3)[-1]
    else:
        exp = full_name.split("_", 1)[0]
        mkn = full_name.split("_", 1)[1]

    print(mkn, exp)

    gzf = gzip.open(path)
    d = json.load(gzf)

    xla_ops_list = []
    tf_ops_list = []
    tf_name_list = []
    dur_list = []

    cpu = 0
    tpu = 0
    for event in d['traceEvents']:
        if 'pid' in event and event['pid'] == 3:
            # Event is on TPU.
            tpu += 1
            if 'tid' in event:
                if event['tid'] == 5 and 'dur' in event:
                    # print(event)
                    # XLA Ops
                    if event["args"]["hlo_category"] == "non-fusion elementwise":
                        continue
                    dur_list += [event['dur']]

        else:
            cpu += 1

    if mkn not in result:
        result[mkn] = {}
    result[mkn][exp] = sum(dur_list[3:])/len(dur_list[3:])
    print(dur_list)

for root, subdirs, files in os.walk("exp_logs"):
    # print(root, subdirs, files)
    for file in files:
        if "trace.json.gz" in file:
            parse(os.path.join(root, file))

print("=====================================")
print()

exp_list = ["gemm", "gemm_and_norm", "norm"]

print("b,m,k,n,gemm,gemm_and_norm,norm")
for mkn in result:
    print(mkn.replace("_", ","), end=",")
    for exp in exp_list:
        if not exp in result[mkn]:
            print("", end=",")
            continue
        print(result[mkn][exp], end=",")
    print()
