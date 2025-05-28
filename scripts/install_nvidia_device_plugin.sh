#!/bin/bash

# Load env data
source .env

# Apply manifests
kubectl apply -f "${SOURCE_KUBERNETES_PATH}/manifests/runtime-class-nvidia.yaml"

# Add NVIDIA helm repo and update
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update

# Install NVIDIA device plugin
helm install nvidia-device-plugin nvdp/nvidia-device-plugin --set=runtimeClassName=nvidia --create-namespace --namespace nvidia-device-plugin --version 0.13.0

