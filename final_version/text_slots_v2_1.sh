#!/usr/bin/env bash

set -euo pipefail

exec /opt/anaconda3/envs/agent/bin/python -u /Users/ken/MM/Pipeline/final_version/pics/closed_loop.py \
  --config /Users/ken/MM/Pipeline/final_version/closed_loop.config.yaml \
  --api-timeout 180
