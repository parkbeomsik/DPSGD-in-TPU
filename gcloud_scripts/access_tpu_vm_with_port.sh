export PROJECT_ID=dpsgd-347112
export ZONE=us-central1-b


gcloud alpha compute tpus tpu-vm ssh gemm-test \
   --project ${PROJECT_ID} \
   --zone ${ZONE} \
   --ssh-flag="-4 -L 6006:localhost:6006"