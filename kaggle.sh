#!/bin/bash
curl -L -o metr-la-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/annnnguyen/metr-la-dataset

unzip metr-la-dataset.zip -d metr-la-dataset

mkdir -p logs