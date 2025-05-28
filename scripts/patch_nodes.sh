#!/bin/bash

# Load env data
source .env

### Script start

# Change to repo root
cd "${REPO_ROOT_PATH}"

# Apply controlplane patches
for i in ${!CONTROL_PLANE_IPS[@]}; do
  for patch in $(ls "${CONFIG_PATCHES_PATH}/control$((i + 1))"); do
    echo "Applying '${patch}' ..."
    talosctl patch mc -n ${CONTROL_PLANE_IPS[$i]} --patch-file "${CONFIG_PATCHES_PATH}/control$((i + 1))/${patch}"
  done
done

# Apply worker patches
for i in ${!WORKER_IPS[@]}; do
  for patch in $(ls "${CONFIG_PATCHES_PATH}/worker$((i + 1))"); do
    echo "Applying '${patch}' ..."
    talosctl patch mc -n ${WORKER_IPS[$i]} --patch-file "${CONFIG_PATCHES_PATH}/worker$((i + 1))/${patch}"
  done
done

# Wait for install and reboot
read -p "Wait for nodes to come up and get into ready state, then press enter to continue"

