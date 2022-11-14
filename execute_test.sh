export PROJECT_ID=dpsgd-342511
export ZONE=us-central1-b

# gcloud alpha compute tpus tpu-vm scp --recurse ./scripts gemm-test:~/.

gcloud alpha compute tpus tpu-vm ssh gemm-test \
--project ${PROJECT_ID} \
--zone ${ZONE} \
--command "./scripts/layer_wise_gemm_test_conv.sh"