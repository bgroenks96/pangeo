# Google project id
export PROJECTID='thesis-research-255223'
# Kubernetes cluster admin
export EMAIL='bgroe8@gmail.com'

# Set up zone and region (see: https://cloud.google.com/compute/docs/regions-zones/)
export ZONE='us-west1-b'
export REGION='us-west1'

# cluster size settings: modify as needed to fit your needs / budget
export NUM_DEFAULT_NODES=2
export MIN_DEFAULT_NODES=1
export MAX_DEFAULT_NODES=2
export MIN_WORKER_NODES=0
export MAX_WORKER_NODES=20
export MIN_GPU_WORKER_NODES=0
export MAX_GPU_WORKER_NODES=2
export CLUSTER_NAME='pangeo-ml'

# https://cloud.google.com/compute/pricing
# change the machine typer based on your computing needs
export WORKER_MACHINE_TYPE='n1-standard-8'
export GPU_WORKER_MACHINE_TYPE='n1-highmem-4'
export GPU_COUNT=1
export GPU_TYPE='nvidia-tesla-k80'
