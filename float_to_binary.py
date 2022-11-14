import random
import numpy as np
import torch

import struct
def bfloat16(num):
    return torch.Tensor([num]).to(torch.bfloat16).item()

def float32(num):
    return torch.Tensor([num]).to(torch.float32).item()

def bfloat16_mult(lhs, rhs):
    return (torch.Tensor([lhs]).to(torch.bfloat16) * torch.Tensor([rhs]).to(torch.bfloat16)).item()

def fp32_bf16_add(lhs, rhs):
    return (torch.Tensor([lhs]).to(torch.float32) + torch.Tensor([rhs]).to(torch.bfloat16)).item()

def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def mult_test_cases(lhs, rhs):
    b_lhs = binary(bfloat16(lhs))[:16]
    assert(binary(bfloat16(lhs))[16:] == "0"*16)
    b_rhs = binary(bfloat16(rhs))[:16]
    assert(binary(bfloat16(rhs))[16:] == "0"*16)
    b_ret = binary(bfloat16_mult(lhs, rhs))[:16]
    assert(binary(bfloat16_mult(lhs, rhs))[16:] == "0"*16)
    print(f"'{{16'b{b_lhs}, 16'b{b_rhs}, 16'b{b_ret}}},")

def add_test_cases(lhs, rhs):
    b_lhs = binary(float32(lhs))
    b_rhs = binary(bfloat16(rhs))
    assert(binary(bfloat16(rhs))[16:] == "0"*16)
    b_ret = binary(fp32_bf16_add(lhs, rhs))
    # print(float32(lhs), bfloat16(rhs), fp32_bf16_add(lhs, rhs))
    print(f"'{{32'b{b_lhs}, 32'b{b_rhs}, 32'b{b_ret}}},")

def fma_test_cases(i, w, ps):
    b_i = binary(bfloat16(i))
    b_w = binary(bfloat16(w))
    b_ps = binary(float32(ps))
    assert(binary(bfloat16(rhs))[16:] == "0"*16)
    b_ret = binary(fp32_bf16_add(ps, bfloat16_mult(i, w)))
    # print(float32(lhs), bfloat16(rhs), fp32_bf16_add(lhs, rhs))
    print(f"'{{32'b{b_i}, 32'b{b_ps}, 32'b{b_ret}}},")

lhs = 1.5
rhs = -3.625

ret = binary(lhs)
print("lhs : ", lhs)
print("------- BF16 --------")
print(ret[:16])
print(ret[0], ret[1:9], ret[10:16])
print("------- FP32 --------")
print(ret)
print(ret[0], ret[1:9], ret[10:])
print("")

ret = binary(rhs)
print("rhs : ", rhs)
print("------- BF16 --------")
print(ret[:16])
print(ret[0], ret[1:9], ret[10:16])
print("------- FP32 --------")
print(ret)
print(ret[0], ret[1:9], ret[10:])
print("")

ret = binary(lhs*rhs)
print("lhs*rhs : ", lhs*rhs)
print("------- BF16 --------")
print(ret[:16])
print(ret[0], ret[1:9], ret[10:16])
print("------- FP32 --------")
print(ret)
print(ret[0], ret[1:9], ret[10:])
print("")


mult_test_cases(-1.3984375, 1.5)

# for i in range(100):
#     lhs = np.float32((random.random() - 0.5) * 100)
#     rhs = np.float32((random.random() - 0.5) * 100)
#     add_test_cases(lhs, rhs)

for i in range(100):
    i = np.float32((random.random() - 0.5) * 100)
    w = np.float32(1.5)
    ps = np.float32((random.random() - 0.5) * 100)
    fma_test_cases(i, w, ps)