#!/bin/bash

# Load env data
source .env

### Script start

# Change to repo root
cd "${REPO_ROOT_PATH}"

# Generate config dir
mkdir -p "${CONFIG_PATH}"

# Generate talos secrets
talosctl gen secrets --output-file "${CONFIG_SECRETS_PATH}"

