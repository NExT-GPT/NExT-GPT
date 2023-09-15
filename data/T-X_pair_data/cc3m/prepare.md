

## Preparation

1. Download `Train_GCC-training.tsv` dataset from [here](https://ai.google.com/research/ConceptualCaptions/download) 


2. Download images via the following commands:
```commandline
pip install img2dataset

img2dataset --url_list Train_GCC-training.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder cc3m --processes_count 16 --thread_count 64 --image_size 256 --enable_wandb True
```
Note that: 

- `url_list` A file with the list of url of images to download. It can be a folder of such files. (required)
- `image_size` The size to resize image to (default 256)
- `output_folder` The path to the output folder. (default "images")
- `processes_count` The number of processes used for downloading the pictures. This is important to be high for performance. (default 1)
- `thread_count` The number of threads used for downloading the pictures. This is important to be high for performance. (default 256)
- `output_format` decides how to save pictures (default files)
  - `files saves` as a set of subfolder containing pictures
  - `webdataset` saves as tars containing pictures
  - ...
- `url_col` the name of the url column for parquet and csv (default url)
- `caption_col` the name of the caption column for parquet and csv (default None)
- `enable_wandb` whether to enable wandb logging (default False)

For more details, please refer to [img2dataset](https://github.com/rom1504/img2dataset/blob/main/README.md)



3. Once you've downloaded the dataset, please verify the download status, as some image files may not have been successfully downloaded. Afterward, organize the dataset into a json file with the following format:

```angular2html
[   
    {
        "caption": "pitcher pitches against sports team",
        "image_name": "000002362.jpg"
    },
    {
        "caption": "sea beach with mountains on the horizon , yellow umbrella and sand",
        "image_name": "000007816.jpg"
    },
    ...
]
```
The data file structure should be:
```angular2html
data/T-X_pair_data/cc3m
├── cc3m.json
├── images
|   ├── 000002362.jpg
|   ├── 000007816.jpg
|   └── ...
```