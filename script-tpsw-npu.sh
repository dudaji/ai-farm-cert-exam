#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $1 <results_suffix>"
    exit 1
fi
results_suffix=$1

request_rate=16
num_prompts=10000

results_dir="$(pwd)/data/results-$results_suffix"
if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"
fi

PRETRAINED_ID=$(curl localhost:8000/v1/models | jq -r ".data[0].id")
VLLM_DIR="$(pwd)/vllm"
dataset_path="$(pwd)/data/inputs/sharegpt_vi-en.json"

cp benchmark_serving.py $VLLM_DIR/benchmarks/benchmark_serving.py
cd $VLLM_DIR
python benchmarks/benchmark_serving.py --backend vllm \
    --model $PRETRAINED_ID \
    --port 8000 \
    --dataset-name sharegpt \
    --dataset-path $dataset_path \
    --request-rate $request_rate \
    --num-prompts $num_prompts \
    --result-dir $results_dir \
    --metric-percentiles "25,50,75,90,95,99" \
    --enable-device-monitor "npu" \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --save-result 

cd -
