import plotly.figure_factory as ff
import numpy as np
import math

hist_data_m = [[] for _ in range(4)]
hist_data_k = [[] for _ in range(4)]
hist_data_n = [[] for _ in range(4)]

# Add histogram data
with open("test.csv", "r") as f:
    while True:
        line = f.readline()
        cells = line.replace("\n", ",").split(",")
        if len(cells) == 1:
            break

        if cells[0] == "FW":
            index = 0
        elif cells[0] == "BW_data":
            index = 1
        elif cells[0] == "BW_weight_batch":
            index = 2
        else:
            index = 3

        print(index, hist_data_m, cells)
        hist_data_m[index].append(math.log(int(cells[1]), 10))
        hist_data_k[index].append(math.log(int(cells[2]), 10))
        hist_data_n[index].append(math.log(int(cells[3]), 10))
        
        if not line:
            break

group_labels = ['FW', 'BW_data', 'BW_weight_batch', 'BW_weight_sample']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data_k, group_labels, bin_size = .2)
fig.show()