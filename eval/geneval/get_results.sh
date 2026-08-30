#!/bin/bash

MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 Mobile-O-0.5B"
    exit 1
fi

python eval/geneval/evaluation/summary_scores.py "eval/geneval/${MODEL_NAME}/results.jsonl"


