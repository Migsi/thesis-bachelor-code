#!/bin/bash

# Setup talos kubernetes cluster and prepare it for torchX GPU accelerated workloads

### Script start

# Check if env data is set up
if [ ! -f .env ]; then
  echo "Environment config missing! Ensure you run this script from the repository root."
  echo "If you didn't generate a config yet, do so by creating a copy of 'template.env'"
  echo "named '.env' and adjusting it's contents accordingly."
  exit -1
fi

# Load env data
source .env

# Change to repo root
cd "${REPO_ROOT_PATH}"

echo "Throughout the setup you will be asked to 'wait for the nodes to come up and get into ready state'"
echo "multiple times. To do so, open a second terminal, navigate to the repository root and run"
echo "'talosctl dashboard --talosconfig \"${CONFIG_PATH}/talosconfig\"'."

read -p "Press enter to continue"

# Run all setup scripts
bash "${SCRIPT_PATH}/generate_secrets.sh"
bash "${SCRIPT_PATH}/generate_config.sh"
bash "${SCRIPT_PATH}/install_nodes.sh"
bash "${SCRIPT_PATH}/patch_nodes.sh"
bash "${SCRIPT_PATH}/bootstrap_cluster.sh"

# Load config data
source source_me.sh

echo "Your cluster should be set up now! To check check the state of the cluster, you can now"
echo "source the script 'source_me.sh' and run 'talosctl dashboard' without parameters."

# Wait for cluster to become ready
read -p "Wait for nodes to come up and get into ready state, then press enter to continue"

# Install required manifests
bash "${SCRIPT_PATH}/install_nvidia_device_plugin.sh"
bash "${SCRIPT_PATH}/install_volcano.sh"

echo "Setup done! To interact with the cluster, run 'source source_me.sh' once per shell."

