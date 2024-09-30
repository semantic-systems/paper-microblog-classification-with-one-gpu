#!/bin/bash
#SBATCH --gpus-per-node=2

huggingface-cli login --token $HFTOKENS
python main.py ./sequence_classifier/configs/tweeteval/temperature/09/bert-base-uncased/offensive.yaml
