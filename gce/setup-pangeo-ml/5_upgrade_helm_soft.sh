#!/bin/bash

set -e

helm upgrade pangeohub local/pangeo-ml \
   -f secret_config.yaml \
   -f jupyter_config.yaml
