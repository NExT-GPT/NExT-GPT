#!/bin/bash

# wget -nc http://www.robots.ox.ac.uk/~maxbain/webvid/results_10M_train.csv

video2dataset --url_list="results_2M_train.csv" \
        --input_format="csv" \
        --output-format="webdataset" \
	--output_folder="dataset" \
        --url_col="contentUrl" \
        --caption_col="name" \
        --save_additional_columns='[videoid,page_idx,page_dir,duration]' \
        --enable_wandb=True \
	--config="config.yaml" \