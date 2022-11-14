# for i in range(64):
#     print(f"FP32_ADD fp32_add_{i} (c[DATA_SIZE_32*{i+1}-1:DATA_SIZE_32*{i}], out_en, a[DATA_SIZE_32*{2*i+2}-1:DATA_SIZE_32*{2*i+1}], a[DATA_SIZE_32*{2*i+1}-1:DATA_SIZE_32*{2*i}], in_en, clk, rst_n, set);")

for i in range(128):
    print(f"test_cases[{i}],")