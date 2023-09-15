
## Preparation


### Requirements

Python 3.9 (it may work with other versions, but it has not been tested)

### Installation

```angular2html
# Install ffmpeg
sudo apt install ffmpeg
# Install audiocaps-download
pip install audiocaps-download
```

### Usage
- Download `csv` file in [here](https://audiocaps.github.io/). The header of the CSV file are:
```angular2html
audiocap_id,youtube_id,start_time,caption
```

- Download audio by following codes:
```angular2html
from audiocaps_download import Downloader
d = Downloader(root_path='data/T-X_pair_data/audiocap/', n_jobs=16)
d.download(format = 'wav')
```


The main class is `audiocaps_download.Downloader`. It is initialized using the following parameters:

- `root_path`: the path to the directory where the dataset will be downloaded.
- `n_jobs`: the number of parallel downloads. Default is 1.
The methods of the class are:

- `download(format='vorbis', quality=5)`: downloads the dataset.
- The format can be one of the following (supported by `yt-dlp` `--audio-format parameter`):
  - `vorbis`: downloads the dataset in Ogg Vorbis format. This is the default.
  - `wav`: downloads the dataset in WAV format.
  - `mp3`: downloads the dataset in MP3 format.
  - `m4a`: downloads the dataset in M4A format.
  - `flac`: downloads the dataset in FLAC format.
  - `opus`: downloads the dataset in Opus format.
  - `webm`: downloads the dataset in WebM format.
  - ... and many more. 
  - The quality can be an integer between 0 and 10. Default is 5.
- `load_dataset()`: reads the csv files from the original repository. It is not used externally.
- `download_file(...)`: downloads a single file. It is not used externally.

### Postprocess
Once you've downloaded the dataset, please verify the download status, as some audio files may not have been successfully downloaded. Afterward, organize the dataset into a json file with the following format:
```angular2html
[   
    {
        "caption": "A woman talks nearby as water pours",
        "audio_name": "91139.wav"
    },
    {
        "caption": "The wind is blowing and rustling occurs",
        "audio_name": "11543.wav"
    },
    ...
]
```
The data file structure should be:
```angular2html
data/T-X_pair_data/audiocap
├── audiocap.json
├── audios
|   ├── 91139.wav
|   ├── 11543.wav
|   └── ...
```
