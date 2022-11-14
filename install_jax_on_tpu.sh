export PROJECT_ID=dpsgd-342511
export ZONE=us-central1-b

gcloud alpha compute tpus tpu-vm ssh gemm-test \
--project ${PROJECT_ID} \
--zone ${ZONE} \
--command "pip install 'jax[tpu]>=0.2.16' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"