#!/bin/bash

# Sets up virtual environment

# Avoid script is run instead of sourced
if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  echo "This script is meant to be sourced only and can not be run."
  exit 255
fi

### Variables
path_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

name_venv="torchx-demo"
path_venv=".venv/${name_venv}"

path_activate="${path_project_root}/${path_venv}/bin/activate"
path_requirements="${path_project_root}/requirements.txt"

### Script starts here

# Detect if running inside a venv, deactivate if true
python -c 'import sys; exit (1 if sys.prefix != sys.base_prefix else 0)'
exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo
  echo "Deactivating currently active virtual environment ..."
  echo
  deactivate
fi

# Change to project root directory
cd "${path_project_root}" || exit 254

rm -rf "${path_venv}"
exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo "Failed to delete existing directory for virtual environment"
  return 254
fi

echo
echo "Creating virtual environment ..."
echo

mkdir -p "${path_venv}"
exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo "Failed to create directory for virtual environment"
  return 254
fi

output="$(virtualenv --clear "${path_venv}" 2>&1)"
exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo "Failed to create virtual environment: ${output}"
  return 254
fi

echo
echo "Installing dependencies ..."
echo

# shellcheck disable=SC1090
source "${path_activate}"

pip install -r "${path_requirements}"
exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo "Failed to install dependencies"
  return 254
fi

echo
echo "Setup done!"
echo