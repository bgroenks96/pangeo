#!/bin/bash

set -e

helm upgrade pangeohub pangeo/pangeo \
   -f secret_config.yaml \
   -f jupyter_config.yaml
