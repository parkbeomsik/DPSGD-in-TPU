# DPSGD-in-TPU
A bunch of scripts to measure the performance of Differentially Private SGD (DP-SGD) on TPU. It also includes scripts to measure the performance of key GEMMs in DP-SGD.

## Description
### gcloud_scripts/
Bash shell scripts to make the use of Google Cloud TPU easy. It helps to access, upload, and download files in Google Cloud TPU instance.

### diva_tpu_utilization_scripts/
Bash shell and python scripts to get a TPU utilization in "DiVa: An Accelerator for Differentially Private Machine Learning" (https://arxiv.org/abs/2208.12392).

### tpu_exp/
Python scripts to measure the runtime of GEMM and computing norm in TPU.

### tutorials/, tutorials_new/
Bash shell and python scripts to get an end-to-end performance of DP-SGD in TPU. You can execute the experiments using bash shell script.

