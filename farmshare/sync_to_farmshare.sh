#!/bin/bash
# Push local code/ + scripts/ to FarmShare marc/. Heavy artifacts
# (data/results/gifs/logs/envs) stay FarmShare-only and are NOT synced back.
set -e
LOCAL=/Users/frozone/Documents/MARC
REMOTE=flukol@rice.stanford.edu:/scratch/users/flukol/marc

rsync -avz --delete \
  --exclude '__pycache__' --exclude '.DS_Store' --exclude 'JaxMARL_ref' \
  "$LOCAL/code/" "$REMOTE/code/"
rsync -avz --exclude '.DS_Store' "$LOCAL/scripts/" "$REMOTE/scripts/"
echo "synced code/ + scripts/ -> FarmShare (JaxMARL_ref preserved remotely)"
