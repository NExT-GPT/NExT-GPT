
# Data Preparation


## Alignment Pretraining Data

#### Audiocap

To begin, download and preprocess the original dataset as outlined in  [prepara.md](T_X_pair_data/audiocap/prepare.md). This will yield the following directory and files:
```angular2html
data/T-X_pair_data/audiocap
├── audiocap.json
├── audios
|   ├── 91139.wav
|   ├── 11543.wav
|   └── ...
```
Next, run [prepare_data.py](prepare_data.py) to preprocess the Audiocap data for encoding-side audio comprehension and decoding-side audio generation. This process will add additional JSON files for comprehension and generation purposes, as shown below:
```angular2html
data/T-X_pair_data/audiocap
├── audiocap.json
├── audiocap_comprehension.json
├── audiocap_generation.json
├── audios
|   ├── 91139.wav
|   ├── 11543.wav
|   └── ...
```
----------------

#### CC3M

Firstly, download and preprocess the original dataset as outlined in [prepare.md](T_X_pair_data/cc3m/prepare.md). This will yield the following directory and files:
```angular2html
data/T-X_pair_data/cc3m
├── cc3m.json
├── images
|   ├── 001315438.jpg
|   ├── 001313716.jpg
|   └── ...
```
Then, run [prepare_data.py](prepare_data.py) to preprocess the CC3M data for encoding-side image comprehension and decoding-side image generation.

```angular2html
data/T-X_pair_data/cc3m
├── cc3m.json
├── cc3m_comprehension.json
├── cc3m_generation.json
├── images
|   ├── 001315438.jpg
|   ├── 001313716.jpg
|   └── ...
```


---------------

#### webvid
Firstly, download and preprocess the original dataset as outlined in [prepare.md](T_X_pair_data/webvid/prepare.md). This will yield the following directory and files:
```angular2html
data/T-X_pair_data/webvid
├── webvid.json
├── videos
|   ├── 31353427.mp4
|   ├── 14781349.mp4
|   └── ...
```
Or just run the following command:
```
bash download.sh  # download the dataset
bash preprocess.sh  # uncompress the dataset
```

Then, run [prepare_data.py](prepare_data.py) to preprocess the Webvid data for encoding-side video comprehension and decoding-side video generation:
```angular2html
data/T-X_pair_data/webvid
├── webvid.json
├── webvid_comprehension.json
├── webvid_generation.json
├── videos
|   ├── 31353427.mp4
|   ├── 14781349.mp4
|   └── ...
```

------------



## Instruction Data

#### T-T+X Data
We provide the construction method in [construction directory](IT_data/T-T+X_data/construction), please refer to it for details. 
Then, run the [prepare_data.py](prepare_data.py) to preprocess the dataset format for instruction-tuning:
```
[   
    # audio instruction dataset
    {
        "audio_captions": [
            "A vehicle engine works at regular pace and a man speeches"
        ],
        "audio_caption_embeddings": [
            "34706.wav.npy"
        ],
        "conversations": [
            {
                "from": "human",
                "value": "I'm working on a project that requires the sound of a vehicle engine working at a regular pace, along with a man giving a speech. Can you assist me with that?",
                "input_modality": "text"
            },
            {
                "from": "gpt",
                "value": "Absolutely! I have just the audio clip you need. Imagine yourself transported to the scene: the steady hum of a vehicle engine reverberates in the background, creating an ambient backdrop. Amidst this mechanical symphony, a man's voice takes center stage, delivering an engaging and inspiring speech that captivates the listener's attention. <audio>",
                "caption": "A vehicle engine works at regular pace and a man speeches",
                "output_modality": "audio"
            }
        ]
    },
    ...
    # image instruction dataset
    {
        "image_captions": [
            "gray wolf , keeper rollicking about in the snow with a socialised"
        ],
        "image_caption_embeddings": [
            "it_img_0.jpg.npy"
        ],
        "conversations":[...]

    },
    ...
    # video instruction dataset
    {
        "video_captions": [
            "Middle aged woman mother doing sport yoga exercises together with her little son toddler boy at home. healthy lifestyle"
        ],
        "video_caption_embeddings": [
            "it_vid_0.mp4.npy"
        ],
        "conversations": [...]
    },
    ...    
]
```

#### MosIT Data
We provide the detailed prompts for constructing and preprocessing the MosIT dataset, kindly refer to [construction directory](IT_data/MosIT_data/construction).
The data format is as follows:
```
[
    {
        "image_list": [
            {
                "image_name": "mosit_i_0000000_00.jpg",
                "data_source": "mosit",
                "caption": "A visual representation of the rule of thirds"
            },
            {
                "image_name": "mosit_i_0000000_01.jpg",
                "data_source": "mosit",
                "caption": "An image of a flowing waterfall captured with a long exposure"
            },
            {
                "image_name": "mosit_i_0000000_02.jpg",
                "data_source": "mosit",
                "caption": "A photo of the Sony Alpha a6000 with a compact lens attached"
            }
        ],
        "video_list": [],
        "audio_list": [],
        "conversation": [...]
    },
    ...
]
```
Then, run the [prepare_data.py](prepare_data.py) to preprocess the dataset format for instruction-tuning:
```
[
    {
        "image_captions": [
            "An image of a basketball court",
            "A diagram showcasing different basketball drills",
            "An image of a player passing the basketball"
        ],
        "image_caption_embeddings": [
            "mosit_i_0000002_00.jpg.npy",
            "mosit_i_0000002_01.jpg.npy",
            "mosit_i_0000002_04.jpg.npy"
        ],
        "conversations": [...]
    },
    ...
]
``` 

### T+X-T Data

We also leverage [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), [llava-150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) and [videochat](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT) for instruction fine-tuning. 







