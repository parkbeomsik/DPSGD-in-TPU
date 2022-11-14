gcloud alpha compute tpus tpu-vm scp ./layer_wise_gemm_test_conv.py gemm-test:~/.
gcloud alpha compute tpus tpu-vm scp ./layer_wise_gemm_test_conv.sh gemm-test:~/.
gcloud alpha compute tpus tpu-vm scp ./layer_wise_gemm_test_transformer.py gemm-test:~/.
gcloud alpha compute tpus tpu-vm scp ./layer_wise_gemm_test_transformer.sh gemm-test:~/.

for entry in layer_configs/*mm.csv
do
  echo $entry
  gcloud alpha compute tpus tpu-vm scp $file gemm-test:$file
done