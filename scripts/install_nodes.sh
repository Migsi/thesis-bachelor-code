#!/bin/bash

# Load env data
source .env

### Script start

# Change to repo root
cd "${REPO_ROOT_PATH}"

# Install controlplane
for i in ${!CONTROL_PLANE_IPS[@]}; do
  talosctl apply-config --insecure --nodes ${CONTROL_PLANE_IPS[$i]} --file "${CONFIG_PATH}/controlplane.yaml" --config-patch "@${CONFIG_PATCHES_PATH}/control$((i + 1))/extra-kernel-args.yaml"
done

# Install worker
for i in ${!WORKER_IPS[@]}; do
  talosctl apply-config --insecure --nodes ${WORKER_IPS[$i]} --file "${CONFIG_PATH}/worker.yaml" --config-patch "@${CONFIG_PATCHES_PATH}/worker$((i + 1))/extra-kernel-args.yaml"
done

# Wait for install and reboot
read -p "Wait for nodes to come up and get into ready state, then press enter to continue"

