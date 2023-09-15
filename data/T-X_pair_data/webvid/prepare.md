## Preparation

WebVid is a large-scale text-video dataset, containing 10 million video-text pairs scraped from the stock footage sites.
To download the dataset, run the following command:

```angular2html
wget -nc http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_train.csv

video2dataset --url_list="results_10M_train.csv" \
        --input_format="csv" \
        --output-format="webdataset" \
	    --output_folder="dataset" \
        --url_col="contentUrl" \
        --caption_col="name" \
        --save_additional_columns='[videoid,page_idx,page_dir,duration]' \
        --enable_wandb=True \
	    --config="path/to/config.yaml" \
```
For more datails, please refer to [video2dataset](https://github.com/iejMac/video2dataset/blob/main/dataset_examples/WebVid.md).


### Postprocess
Once you've downloaded the dataset, please verify the download status, as some video files may not have been successfully downloaded. Afterward, organize the dataset into a json file with the following format:
```angular2html
[   
    {
        "caption": "Merida, mexico - may 23, 2017: tourists are walking on a roadside near catholic church in the street of mexico at sunny summer day.",
        "video_name": "31353427.mp4"
    },
    {
        "caption": "Happy family using laptop on bed at home",
        "video_name": "14781349.mp4"
    },
    ...
]
```

The data file structure should be:
```angular2html
data/T-X_pair_data/webvid
├── webvid.json
├── videos
|   ├── 31353427.mp4
|   ├── 14781349.mp4
|   └── ...
```