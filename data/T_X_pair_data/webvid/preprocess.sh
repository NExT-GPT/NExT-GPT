#!/bin/bash

# in this file, we will unzip the downloaded data and preprocess it
D_NUM=0
for i in {100..226}; do
    if [ $D_NUM -lt 10 ]; then
        rm -f ./data/T_X_pair_data/webvid/videos/*.txt
        D_NUM=1
    else
        D_NUM=$((D_NUM + 1))
    fi
    echo "Extracting 00${i}.tar"
      
        tar -xf ./data/T_X_pair_data/webvid/dataset/00${i}.tar -C ./data/T_X_pair_data/webvid/videos/
    done