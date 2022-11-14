import gzip
import json
from turtle import update
import os


def parse(path):
    print("----------------------------------------")
    print(path[:path.find("2022")].split("/")[1])

    gzf = gzip.open(path)
    d = json.load(gzf)

    xla_ops_list = []
    tf_ops_list = []
    tf_name_list = []

    cpu = 0
    tpu = 0
    for event in d['traceEvents']:
        if 'pid' in event and event['pid'] == 3:
            # Event is on TPU.
            tpu += 1
            if 'tid' in event:
                if event['tid'] == 5:
                    # XLA Ops
                    xla_ops_list += [event]
                elif event['tid'] == 3:
                    # TF Ops
                    tf_ops_list += [event]
                elif event['tid'] == 2:
                    tf_name_list += [event]

        else:
            cpu += 1

    forward = 0
    backward = 0
    computing_norms = 0
    clipping_grads = 0
    second_backprop = 0
    add_noise_and_reduce = 0
    update_params = 0
    elses = 0


    for event in xla_ops_list:
        anyone = False
        if 'ts' in event and 'while' not in event['name']:
            anyone = False
            for tf_name in tf_name_list:
                if 'ts' in tf_name:
                    # print(tf_name)
                    if (tf_name['ts'] <= event['ts'] + 0.1) and (event['ts'] + event['dur'] - 0.1 <= tf_name['ts'] + tf_name['dur']):
                        # print(event['ts'], event['dur'], tf_name['ts'], tf_name['dur'])
                        if 'tf_name' not in event:
                            event['tf_name'] = [tf_name]
                        else:
                            event['tf_name'] += [tf_name] 
                        
                        if "_SGD_" not in path:
                            if tf_name['args']['l'] == '1' and tf_name['name'] == "computing_grads":
                                forward += event['dur']
                                anyone=True
                            elif tf_name['args']['l'] == '2' and tf_name['name'] == "computing_grads":
                                backward += event['dur']
                                anyone=True
                            elif tf_name['args']['l'] in ['0', '1', '2'] and tf_name['name'] == "second_backprop":
                                second_backprop += event['dur']
                                anyone=True
                        else:
                            if tf_name['args']['l'] == '0' and tf_name['name'] == "computing_grads":
                                forward += event['dur']
                                anyone=True
                            elif tf_name['args']['l'] == '1' and tf_name['name'] == "computing_grads":
                                backward += event['dur']
                                anyone=True
                            elif tf_name['args']['l'] in ['0', '1', '2'] and tf_name['name'] == "second_backprop":
                                second_backprop += event['dur']
                                anyone=True
                        if tf_name['args']['l'] == '1' and tf_name['name'] == "computing_norms":
                            computing_norms += event['dur']
                            anyone=True
                        if tf_name['args']['l'] == '1' and tf_name['name'] == "compute_clipping_factor":
                            computing_norms += event['dur']
                            anyone=True
                        elif tf_name['args']['l'] == '1' and tf_name['name'] == "clipping_grads":
                            clipping_grads += event['dur']
                            anyone=True
                        elif tf_name['args']['l'] == '0' and tf_name['name'] == "add_noise_and_reduce":
                            add_noise_and_reduce += event['dur']
                            anyone=True

                        # print(event)
                if anyone==True:
                    break
            #     print(event)

            for tf_op in tf_ops_list:
                if 'ts' in tf_op:
                    # print(tf_op)
                    if (tf_op['ts'] <= event['ts'] + 0.1) and (event['ts'] + event['dur'] - 0.1 <= tf_op['ts'] + tf_op['dur']):
                        if 'tf_op' not in event:
                            event['tf_op'] = [tf_op]
                        else:
                            event['tf_op'] += [tf_op] 

                        if tf_op['name'] == "ResourceApplyGradientDescent":
                            update_params += event['dur']
                            anyone=True

            
            if not anyone:
                # print(event)
                elses += event['dur']

        else:
            anyone = True

        # if anyone:
        #     print(event)
            # print("AAAAAAAAA")

    print("forward,backward,computing_norms,clipping_grads,second_backprop,add_noise_and_reduce,update_params,elses")
    print(forward/1000, end=",")
    print(backward/1000, end=",")
    print(computing_norms/1000, end=",")
    print(clipping_grads/1000, end=",")
    print(second_backprop/1000, end=",")
    print(add_noise_and_reduce/1000, end=",")
    print(update_params/1000, end=",")
    print(elses/1000)

for root, subdirs, files in os.walk("logs"):
    if "large_batch" in root:
        continue
    if "mobilenet" in root:
        continue
    if "squeezenet" in root:
        continue
    # print(root, subdirs, files)
    for file in files:
        if "trace.json.gz" in file:
            parse(os.path.join(root, file))