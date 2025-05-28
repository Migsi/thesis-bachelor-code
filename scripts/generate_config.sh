#!/bin/bash

# Load env data
source .env

### Functions

function copy_cluster_patches() {
  find "${SOURCE_TALOS_PATH}/cluster" -type f -name '*.yaml' | while read -r yaml_file; do
    cp -v "${yaml_file}" "${1}"
  # Translate to JSON while copy
  #  base_name=$(basename "$yaml_file" .yaml)
  #  json_file="${1}/$base_name.json"
  #  echo "'$yaml_file' -> '$json_file'"
  #  yq '.' "$yaml_file" > "$json_file"
  done
}

### Script start

# Change to repo root
cd "${REPO_ROOT_PATH}"

# Generate talos configs
talosctl gen config "${CLUSTER_NAME}" "https://${CONTROL_PLANE_IPS[-1]}:6443" --with-secrets "${CONFIG_SECRETS_PATH}" --install-image "${CONTROL_PLANE_IMAGE_URL}" --output-dir "${CONFIG_PATH}"

# Adjust install image for worker node
sed -i "s#${CONTROL_PLANE_IMAGE_URL}#${WORKER_IMAGE_URL}#g" "${CONFIG_PATH}/worker.yaml"

# Adjust target disk to virtio
sed -i 's#disk: /dev/sda#disk: /dev/vda#g' "${CONFIG_PATH}/controlplane.yaml"
sed -i 's#disk: /dev/sda#disk: /dev/vda#g' "${CONFIG_PATH}/worker.yaml"

# Ensure to wipe disk
sed -i 's#wipe: false#wipe: true#g' "${CONFIG_PATH}/controlplane.yaml"
sed -i 's#wipe: false#wipe: true#g' "${CONFIG_PATH}/worker.yaml"

# Create controlplane patches location and copy patches
for i in $(seq 1 1 "${#CONTROL_PLANE_IPS[@]}"); do
  echo "Creating and copying patches for control${i} ..."
  CONTROL_PATCH_PATH="${CONFIG_PATCHES_PATH}/control${i}"
  mkdir -p "${CONTROL_PATCH_PATH}"
  cp -v "${SOURCE_TALOS_PATH}/machine/extra-kernel-args.yaml" "${CONTROL_PATCH_PATH}"
  cp -v "${SOURCE_TALOS_PATH}/machine/custom-misc.yaml" "${CONTROL_PATCH_PATH}"
  cp -v "${SOURCE_TALOS_PATH}/machine/metrics-kubelet-cert-rotation.yaml" "${CONTROL_PATCH_PATH}"
  yq -iy ".machine.network.hostname = \"control${i}\"" "${CONTROL_PATCH_PATH}/custom-misc.yaml"
  copy_cluster_patches "${CONTROL_PATCH_PATH}"
done

# Create worker patches location and copy patches
for i in $(seq 1 1 "${#WORKER_IPS[@]}"); do
  echo "Creating and copying patches for worker${i} ..."
  WORKER_PATCH_PATH="${CONFIG_PATCHES_PATH}/worker${i}"
  mkdir -p "${WORKER_PATCH_PATH}"
  cp -v "${SOURCE_TALOS_PATH}/machine/extra-kernel-args.yaml" "${WORKER_PATCH_PATH}"
  cp -v "${SOURCE_TALOS_PATH}/machine/custom-misc.yaml" "${WORKER_PATCH_PATH}"
  cp -v "${SOURCE_TALOS_PATH}/machine/metrics-kubelet-cert-rotation.yaml" "${WORKER_PATCH_PATH}"
  cp -v "${SOURCE_TALOS_PATH}/machine/nvidia-gpu-worker.yaml" "${WORKER_PATCH_PATH}"
  cp -v "${SOURCE_TALOS_PATH}/machine/nvidia-default-runtime-class.yaml" "${WORKER_PATCH_PATH}"
  yq -iy ".machine.network.hostname = \"worker${i}\"" "${WORKER_PATCH_PATH}/custom-misc.yaml"
  copy_cluster_patches "${WORKER_PATCH_PATH}"
done

