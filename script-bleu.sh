#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $1 <results_suffix>"
    exit 1
fi
results_suffix=$1

input_path="data/inputs/sharegpt_vi-en.json"

results_dir="data/results-$results_suffix"
if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"
fi

results_path="$results_dir/translation_results_vi-en.json"

python benchmark_bleu.py --input $input_path \
    --src vi \
    --tgt en \
    --output $results_path