#!/bin/bash

# Launches benchmarks shown in thesis

### Variables
BENCHMARK_RUN="4"

RUNS=10
EPOCHS=10
BATCH_SIZE_SINGLE_NODE=128
BATCH_SIZE_MULTI_NODE=64

### Script starts here

# Multi node, DDP
echo
echo "Submitting multi node benchmarks (DDP strategy) ..."
echo

for ((i=0; i < RUNS; i++))
do
  torchx run dist.ddp -j 2x1 --cpu 2 --gpu 1 --memMB 3072 --script dist_resnet18.py -- --storage_path="s3://torchx-playground/run_${BENCHMARK_RUN}" --skip_export --batch_size ${BATCH_SIZE_MULTI_NODE} --strategy="ddp" --epochs=${EPOCHS} &
done
wait

# Single node, DDP
echo
echo "Submitting single node benchmarks (DDP strategy) ..."
echo

for ((i=0; i < RUNS; i++))
do
  torchx run dist.ddp -j 1x1 --cpu 2 --gpu 1 --memMB 3072 --script dist_resnet18.py -- --storage_path="s3://torchx-playground/run_${BENCHMARK_RUN}" --skip_export --batch_size ${BATCH_SIZE_SINGLE_NODE} --strategy="ddp" --epochs=${EPOCHS} &
done
wait

# Single node, FSDP
echo
echo "Submitting single node benchmarks (FSDP strategy) ..."
echo

for ((i=0; i < RUNS; i++))
do
  torchx run dist.ddp -j 1x1 --cpu 2 --gpu 1 --memMB 3072 --script dist_resnet18.py -- --storage_path="s3://torchx-playground/run_${BENCHMARK_RUN}" --skip_export --batch_size ${BATCH_SIZE_SINGLE_NODE} --strategy="fsdp" --epochs=${EPOCHS} &
done
wait

# Multi node, FSDP
echo
echo "Submitting multi node benchmarks (FSDP strategy) ..."
echo

for ((i=0; i < RUNS; i++))
do
  torchx run dist.ddp -j 2x1 --cpu 2 --gpu 1 --memMB 3072 --script dist_resnet18.py -- --storage_path="s3://torchx-playground/run_${BENCHMARK_RUN}" --skip_export --batch_size ${BATCH_SIZE_MULTI_NODE} --strategy="fsdp" --epochs=${EPOCHS} &
done
wait

echo
echo "All jobs submitted!"
