#!/bin/bash

set -e

helm repo add pangeo https://pangeo-data.github.io/helm-chart/
helm repo update

helm install local/pangeo-ml \
   --namespace=pangeo --name=pangeohub  \
   -f secret_config.yaml \
   -f jupyter_config.yaml \
