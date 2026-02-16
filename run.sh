#!/bin/bash
set -e

# Build the image
docker compose build

# Run training (pass any extra args to train_qlora.py)
docker compose run --rm train "$@"
