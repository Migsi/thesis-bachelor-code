#!/bin/bash

# Load env data
source .env

### Script start

# Change to repo root
cd "${REPO_ROOT_PATH}"

# Bootstrap cluster and download kubeconfig to config dir
talosctl bootstrap -n "${CONTROL_PLANE_IPS[-1]}"
talosctl kubeconfig -n "${CONTROL_PLANE_IPS[-1]}" "${CONFIG_PATH}"

