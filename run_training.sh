#!/bin/bash
# exit when any command fails
set -e
dvc pull
python3.9 -u src/models/train_model.py
gsutil -m cp -r outputs gs://fakenewsdetection