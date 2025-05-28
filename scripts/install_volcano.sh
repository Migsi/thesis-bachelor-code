#!/bin/bash

# Load env data
source .env

# Install volcano scheduler
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml

# Install volcano dashboard
# NOTE: currently only local install works, as frontend latest container is not updated by devs
kubectl apply -f "${SOURCE_KUBERNETES_PATH}/manifests/volcano-dashboard.yaml"
#kubectl apply -f https://raw.githubusercontent.com/volcano-sh/dashboard/main/deployment/volcano-dashboard.yaml

# Forward port from localhost to dashboard
# kubectl port-forward svc/volcano-dashboard 8080:80 -n volcano-system --address 127.0.0.1

