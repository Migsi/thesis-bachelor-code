#!/bin/bash

# Check if env data is set up
if [ ! -f .env ]; then
  echo "No environment config found! Ensure you source this script from"
  echo "the repository root and you have run 'setup.sh' first."
  return
fi

# Load env data
source .env

### Script start

export TALOSCONFIG="${CONFIG_PATH}/talosconfig"
export KUBECONFIG="${CONFIG_PATH}/kubeconfig"

talosctl config endpoint "${CONTROL_PLANE_IPS[@]}"
talosctl config node "${CONTROL_PLANE_IPS[@]}" "${WORKER_IPS[@]}"

