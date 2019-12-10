#!/bin/bash

set -e

# create cluster on GCP
gcloud config set project $PROJECTID
gcloud services enable container.googleapis.com #To enable the Kubernetes Engine API

gcloud container clusters create $CLUSTER_NAME --num-nodes=$NUM_DEFAULT_NODES --zone=$ZONE \
    --enable-autoscaling --min-nodes=$MIN_DEFAULT_NODES --max-nodes=$MAX_DEFAULT_NODES  --node-labels=group=core --machine-type=n1-standard-2 \
    --no-enable-legacy-authorization
gcloud container node-pools create worker-pool --zone=$ZONE --cluster=$CLUSTER_NAME --node-labels=group=worker --node-taints=dedicated=worker:NoSchedule \
    --machine-type=$WORKER_MACHINE_TYPE --preemptible --num-nodes=$MIN_WORKER_NODES
gcloud container node-pools create gpu-worker-pool --zone=$ZONE --cluster=$CLUSTER_NAME --node-labels=group=worker \
    --machine-type=$GPU_WORKER_MACHINE_TYPE --num-nodes=$MIN_GPU_WORKER_NODES \
    --accelerator type='nvidia-tesla-v100',count=1
gcloud container clusters update $CLUSTER_NAME --zone=$ZONE --node-pool=worker-pool --enable-autoscaling --max-nodes=$MAX_WORKER_NODES --min-nodes=$MIN_WORKER_NODES
gcloud container clusters update $CLUSTER_NAME --zone=$ZONE --node-pool=gpu-worker-pool --enable-autoscaling --max-nodes=$MAX_GPU_WORKER_NODES --min-nodes=$MIN_GPU_WORKER_NODES
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE --project $PROJECTID
