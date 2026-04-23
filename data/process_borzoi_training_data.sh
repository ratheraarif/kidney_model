#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
data_dir="$REPO_ROOT/resources/"
output="$REPO_ROOT/borzoi_train_data"
PYTHON_BIN="${PYTHON_BIN:-python}"


"$PYTHON_BIN" "$SCRIPT_DIR/basenji_data_h5.py" \
    -b "$data_dir/hg38.blacklist.rep.bed" \
    -g "$data_dir/hg38_gaps.bed" \
    -p 15 \
    -w 32 \
    -l 524288 \
    -v chr2,chr18,chr22 \
    -t chr1 \
    --crop 163840 \
     --stride_train 49173 \
     --stride_test 49173 \
    --local \
    -o "$output" \
    "$data_dir/genome/hg38.ml.fa" \
    "$data_dir/targets_sum.txt"


"$PYTHON_BIN" "$SCRIPT_DIR/process_h5.py" "$output/h5py"


