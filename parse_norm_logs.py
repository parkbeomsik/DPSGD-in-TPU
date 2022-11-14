import gzip
import json
from turtle import update
import os

result = {}

def parse(path):
    print("----------------------------------------")
    print(path[:path.find("2022")].split("/")[1])
    full_name = path[:path.find("2022")].split("/")[1]

    if "only_norm" in full_name:
        exp = "only_norm"
        mkn = full_name.split("_", 2)[-1]
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

for root, subdirs, files in os.walk("norm_logs"):
    # print(root, subdirs, files)
    for file in files:
        if "trace.json.gz" in file:
            parse(os.path.join(root, file))

print("=====================================")
print()

exp_list = ["only_norm"]

with open("parse_norm_logs_results.csv", "w") as f:
    f.write("params,norm\n")
    for mkn in result:
        f.write(f"{mkn},")
        for exp in exp_list:
            if not exp in result[mkn]:
                f.write("\n")
                continue
            f.write(f"{result[mkn][exp]},")
        f.write("\n")
