# <img src="NExT-GPT-Lagacy/code/nextgpt.png" style="width: 5%"> NExT-GPT: Any-to-Any Multimodal LLM
[Shengqiong Wu](https://chocowu.github.io/), [Hao Fei](http://haofei.vip/)*, [Leigang Qu](#), [Wei Ji](https://jiwei0523.github.io/), and [Tat-Seng Chua](https://www.chuatatseng.com/).
(*Correspondence )

**ICML 2024, Oral Paper**

**[NExT++ Research Center](https://www.nextcenter.org/), School of Computing, National University of Singapore**

-----

<a href='https://next-gpt.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='#'><img src='https://img.shields.io/badge/Demo-Page-purple'></a> 
<a href='https://arxiv.org/pdf/2309.05519'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
![License](https://img.shields.io/badge/License-BSD-blue.svg)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=aqw2SCWeWD0)


This repository hosts the code, data and model weight of **NExT-GPT**, the first end-to-end MM-LLM that perceives input and generates output in arbitrary combinations (any-to-any) of text, image, video, and audio and beyond.


**Noted**: we wrap the former old codebase into the [NExT-GPT-Lagacy](NExT-GPT-Lagacy). Please refer to this new codebase for all training and tuning procedures.

-----------

## 🎉 News 

- [x] [2023.09.15] 🚀🚀 Release the code of NExT-GPT in version `7b_tiva_v0`.
- [x] [2023.09.27] 🔨🧩 Added modality-blended batch sampler.
- [x] [2023.10.01] 📢📢 Release the T2M instruction dataset.
- [x] [2023.10.04] 👏👏 Release the checkpoint of NExT-GPT in version [7b_tiva_v0](https://huggingface.co/ChocoWu/nextgpt_7b_tiva_v0) .
- [x] [2023.10.15] 🔨🚀 Update of NExT-GPT in version [7b_tiva_v0](https://huggingface.co/ChocoWu/nextgpt_7b_tiva_v0) .
- [x] [2024.10.07] 👏👏 Release the data and the corresponding construction methods, please refer [DATA_README.md](data/DATA_README.md) for more details.


## 👉 TODO 
- [ ] Updating NExT-GPT in more types&sizes of LLMs.
- [ ] Empowering NExT-GPT with more modalities of inputs&outputs.
- [ ] ...



-----------

## Example Demos
Here we showcase examples generated from NExT-GPT.
For more examples, kindly visit the [webpage](https://next-gpt.github.io/), or the online live [demo](https://acc414b22d6839d28f.gradio.live). 


https://github.com/NExT-GPT/NExT-GPT/assets/18722770/0c2b3d88-a533-4899-ab44-65580fe54538


https://github.com/NExT-GPT/NExT-GPT/assets/18722770/eb1319a6-38aa-4546-a96e-163207e7de93


https://github.com/NExT-GPT/NExT-GPT/assets/18722770/36bec0ad-9bad-4bcf-bc37-92b028f1bc6a



<span id='introduction'/>

## Brief Introduction 


NExt-GPT is built on top of existing pre-trained LLM, multimodal encoder and SoTA diffusion models, with sufficient end-to-end instruction tuning.

<p align="center" width="100%">
<a target="_blank"><img src="figures/framework.png" alt="Video-LLaMA" style="width: 90%; min-width: 200px; display: block; margin: auto;"></a>
</p>

- **Multimodal Encoding Stage.** Leveraging established encoders to encode inputs in various modalities, where these representations are projected into language-like representations comprehensible to the LLM through a projection layer.
- **LLM Understanding and Reasoning Stage.** Harnessing an existing open-sourced LLM as the core to process input information for semantic understanding and reasoning. The LLM not only directly generates text tokens but also produces unique “modality signal” tokens that serve as instructions to dictate the decoding layers whether & what modal content to output correspondingly.
- **Multimodal Generation Stage.** Receiving the multimodal signals with specific instructions from LLM (if any), the Transformer-based output projection layers map the signal token representations into the ones that are understandable to following multimodal decoders.


For more technical details, kindly refer to the [paper](https://arxiv.org/pdf/2309.05519.pdf). 


-----------


<span id='Usage'/>

## Getting Started



<span id='all_catelogue'/>

### Table of Contents:
* <a href='#Code Structure'>1. Code Structure</a>
* <a href='#Environment Preparation'>2. Environment Preparation </a>
* <a href='#Training on Your Own'>3. Training/Adapting NExt-GPT on Your Own</a>
  * <a href='#Prepare Pre-trained Checkpoint'>3.1. Preparing Pre-trained Checkpoint</a>
  * <a href='#Prepare Dataset'>3.2. Preparing Dataset </a>
  * <a href='#Precompute Embeddings'>3.3. Precomputing Embeddings</a>
  * <a href='#Train NExT-GPT'>3.4. Training NExT-GPT</a>
* <a href='#Run NExT-GPT System'>4. Running NExT-GPT System</a>
  * <a href='#Prepare checkpoints'>4.1. Preparing checkpoints</a>
  * <a href='#Deploy Demo System'>4.2. Deploying Demo System</a>
* <a href='#Tuning your own system'>5. Fine-tuning your own System</a>
  * <a href='#Tuning your own dataset'>5.1. Dataset</a>
  * <a href='#Tuning your own framework'>5.2. Model Framework</a>
  * <a href='#Tuning script'>5.3. Fine-tuning</a>
 
****





<span id='Code Structure'/>

### 1. Code Structure 

```
.
|-- NExT-GPT-Lagacy       # the previous version of the model
|-- assets
|-- checkpoints           # save the pretraining and tuning checkpoints
|-- data  
|   |-- IT_data
|   |   |-- MosIT_data
|   |   |-- T+X-T_data    # text+[image/audio/video] to text instruction data
|   |   `-- T-T+X_data    # synthesized text to text+[image/audio/video] instruction data
|   |-- T_X_pair_data     # text-autio pairs data
|   |   |-- audiocap
|   |   |-- cc3m
|   |   `-- webvid
|   |-- embed 
|   `-- prepare_data.py
|-- figures
|-- merge_lora_weights.py
|-- nextgpt
|   |-- __init__.py
|   |-- constants.py
|   |-- conversation.py
|   |-- dataset
|   |   |-- __init__.py
|   |   |-- audio_processor.py
|   |   |-- base_dataset.py
|   |   |-- catalog.py
|   |   |-- concat_dataset.py
|   |   |-- dataset_utils.py
|   |   `-- sampler.py
|   |-- mm_utils.py
|   |-- model
|   |   |-- __init__.py
|   |   |-- apply_delta.py
|   |   |-- builder.py
|   |   |-- consolidate.py
|   |   |-- language_model
|   |   |-- make_delta.py
|   |   |-- multimodal_decoder
|   |   |-- multimodal_encoder
|   |   |-- multimodal_projector
|   |   |-- nextgpt_arch.py
|   |   `-- utils.py
|   `-- utils.py
|-- scripts
|   |-- finetune.sh
|   |-- pretrain_dec.sh
|   |-- pretrain_enc.sh
|   |-- zero2.json
|   |-- zero3.json
|   `-- zero3_offload.json
|-- LICENSE.md
|-- README.md
|-- nextgpt_trainer.py
|-- predict.py
|-- preprocess_embeddings.py
|-- requirements.txt
|-- train.py
|-- train_mem.py
`-- training_utils.py
```


<span id='Environment Preparation'/>


### 2. Environment Preparation  <a href='#all_catelogue'>[Back to Top]</a>
Please first clone the repo and install the required environment, which can be done by running the following commands:
```
conda env create -n nextgpt python=3.8

conda activate nextgpt

# CUDA 12.1
conda install pytorch==2.1.2 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

git clone https://github.com/NExT-GPT/NExT-GPT.git
cd NExT-GPT

pip install -r requirements.txt
```

<span id='Training on Your Own'/>

### 3. Training/Adapting NExt-GPT on Your Own 



<span id='Prepare Pre-trained Checkpoint'/>

#### 3.1. Preparing Pre-trained Checkpoint  <a href='#all_catelogue'>[Back to Top]</a>
NExT-GPT is trained based on following excellent existing models.
Please follow the instructions to prepare the checkpoints.

- `ImageBind`
is the unified image/video/audio encoder. The pre-trained checkpoint can be downloaded from [here](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) with version `huge`. Afterward, put the `imagebind_huge.pth` file at [[.pretrain_ckpt/imagebind]](./pretrain_ckpt/imagebind). 
- `Vicuna`:
prepare the pretrained vicuna from [[here]](https://huggingface.co/lmsys/vicuna-7b-v1.5). Then put the pre-trained model at [[./pretrain_ckpt/vicuna-7b-v1.5/]](./pretrain_ckpt/vicuna-7b-v1.5). 
- `Image Diffusion`
is used to generate images. NExT-GPT uses [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2) with version `
v2`. (_will be automatically downloaded_)
- `Audio Diffusion`
for producing audio content. NExT-GPT employs [AudioLDM](https://github.com/haoheliu/AudioLDM) with version `l-full`. (_will be automatically downloaded_)
- `Video Diffusion`
for the video generation. We employ [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w) with version `v2_576w`. (_will be automatically downloaded_)



<span id='Prepare Dataset'/>

#### 3.2. Preparing Dataset  <a href='#all_catelogue'>[Back to Top]</a>
Please download the following datasets used for model training:

A) T-X pairs data
  - `CC3M` of ***text-image*** pairs, please follow this instruction [[here]](./data/T-X_pair_data/cc3m/prepare.md). Then put the data at [[./data/T-X_pair_data/cc3m]](./data/T-X_pair_data/cc3m).
  - `WebVid` of ***text-video*** pairs, see the [[instruction]](./data/T-X_pair_data/webvid/prepare.md). The file should be saved at [[./data/T-X_pair_data/webvid]](./data/T-X_pair_data/webvid).
  - `AudioCap` of ***text-audio*** pairs, see the [[instruction]](./data/T-X_pair_data/audiocap/prepare.md). Save the data in [[./data/T-X_pair_data/audiocap]](./data/T-X_pair_data/audiocap).

B) Instruction data
  - T+X-T
    - `LLaVA` of the ***visual instruction data***, download it from [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md), and then put it at [[./data/IT_data/T+X-T_data/llava]](./data/IT_data/T+X-T_data/llava/).
    - `Alpaca` of the ***textual instruction data***, download it from [here](https://github.com/tatsu-lab/stanford_alpaca), and then put it at [[./data/IT_data/T+X-T_data/alpaca/]](data/IT_data/T+X-T_data/alpaca/).
    - `VideoChat`, download the ***video instruction data*** [here](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data), and then put it at [[./data/IT_data/T+X-T_data/videochat/]](data/IT_data/T+X-T_data/videochat/).
    
    Side note：After downloading dataset, please run `prepare_data.py` to preprocess the dataset.
  - T-X+T (T2M)
    - The `T-X+T` instruction datasets (T2M) are saved at [[./data/IT_data/T-T+X_data]](./data/IT_data/T-T+X_data).
   
  - MosIT
    - Download the file from [here](), put them in [[./data/IT_data/MosIT_data/]](./data/IT_data/MosIT_data/). (_We are in the process of finalizing the data and handling the copyright issue._) 


<span id='Precompute Embeddings'/>

#### 3.3. Precomputing Embeddings <a href='#all_catelogue'>[Back to Top]</a>
In decoding-side alignment training, we minimize the distance between the representation of signal tokens and captions. 
To save costs of time and memory, we precompute the text embeddings for image, audio and video captions using the text encoder within the respective diffusion models.  

Please run this command before the following training of NExT-GPT, where the produced `embedding` file will be saved at [[./data/embed]](./data/embed).
```angular2html
cd ./code/
python preprocess_embeddings.py ../data/T-X_pair_data/cc3m/cc3m_generation.json image ../data/embed/ stabilityai/stable-diffusion-2
```

Note of arguments:
- args[1]: path of caption file;
- args[2]: modality, which can be `image`, `video`, and `audio`;
- args[3]: saving path of embedding file;
- args[4]: corresponding pre-trained diffusion model name.



<span id='Train NExT-GPT'/>

#### 3.4. Training NExT-GPT  <a href='#all_catelogue'>[Back to Top]</a>

First of all, please refer to the base configuration file [[training_utils.py]](training_utils.py) for the basic system setting of overall modules, and dataset configuration [nextgpt/dataset/catalog.py](nextgpt/dataset/catalog.py).
The whole NExT-GPT training involves 3 steps:

- **Step-1**: Encoding-side LLM-centric Multimodal Alignment. This stage trains the ***input projection layer*** while freezing the ImageBind, LLM, output projection layer.
  ```angular2html
  # Encoding-side LLM-centric Multimodal Alignment
  bash scripts/pretrain_enc.sh
  ```



- **Step-2**: Decoding-side Instruction-following Alignment. This stage trains the ***output projection layers*** while freezing the ImageBind, LLM, input projection layers.
  ```angular2html
  # Encoding-side LLM-centric Multimodal Alignment
  bash scripts/pretrain_enc.sh
  ```





- **Step-3**: Instruction Tuning. This stage instruction-tune 1) the ***LLM*** via LoRA, 2) ***input projection layer*** and 3) ***output projection layer*** on the instruction dataset.
  ```angular2html
  # Encoding-side LLM-centric Multimodal Alignment
  bash scripts/pretrain_enc.sh
  ```




<span id='Run NExT-GPT System'/>

## 4. Running NExT-GPT System <a href='#all_catelogue'>[Back to Top]</a>


<span id='Prepare checkpoints'/>


#### 4.1. Preparing Checkpoints

First, loading the pre-trained NExT-GPT system.
- **Step-1**: load `Frozen parameters`. Please refer to <a href='#Prepare Pre-trained Checkpoint'>3.1 Preparing Pre-trained Checkpoint</a>.

- **Step-2**: load `Tunable parameters`. Please put the NExT-GPT system at [./checkpoints/nextgpt-v1.5-7b](./checkpoints/nextgpt-v1.5-7b). You may either 1) use the params trained yourselves, or 2) download our checkpoints from [Huggingface](). 


#### 4.2. Run the Prediction
Upon completion of the checkpoint loading, you can run the prediction via:
```angular2html
python predict.py
```

---------


<span id='Tuning your own system'/>

## 5. Fine-tuning Your Own System <a href='#all_catelogue'>[Back to Top]</a>


<span id='Tuning your own dataset'>

#### 5.1. Dataset
You can define your own dataset, please refer to the [base_dataset.py](nextgpt/dataset/base_dataset.py), and then add the dataset `catalog` in [catalog.py]([text](nextgpt/dataset/catalog.py)), including the `target` and `parameters`.


<span id='Tuning your own framework'>

#### 5.2. Model Framework
- *Multimodal Encoder*: You can leverage your own multimodal encoder in [multimodal encoder directory](nextgpt/model/multimodal_encoder), and add corresponding code in the [builder.py](nextgpt/model/multimodal_encoder/builder.py).
- *Multimodal Decoder*: You can add your own multimodal decoder, in  [multimodal decoder directory](nextgpt/model/multimodal_decoder), and modify the corresponding code in the [builder.py](nextgpt/model/multimodal_decoder/builder.py).
- *Projector*: You can design your own input and output projector in [multimodal projector](nextgpt/model/multimodal_projector/builder.py).  


<span id='Tuning script'>

#### 5.3. Fine-tuning

You can pre-define the model, data, and training parameters in [training_utils.py](training_utils.py).
Please refer the [finetune.sh](scripts/finetune.sh) for fine-tuning your own model.



---------



## Contact

For any questions or feedback, feel free to contact [Shengqiong Wu](mailto:swu@u.nus.edu) and [Hao Fei](mailto:haofei37@nus.edu.sg).


## Citation

If you find NextGPT useful in your research or applications, please kindly cite:
```
@inproceedings{wu24next,
  title={{NE}x{T}-{GPT}: Any-to-Any Multimodal {LLM}},
  author={Wu, Shengqiong and Fei, Hao and Qu, Leigang and Ji, Wei and Chua, Tat-Seng},
  booktitle={Proceedings of the International Conference on Machine Learning},
  pages = {53366--53397},
  year={2024}
}
```





## Acknowledgements
You may refer to related work that serves as foundations for our framework and code repository, 
[Vicuna](https://github.com/lm-sys/FastChat), 
[ImageBind](https://github.com/facebookresearch/ImageBind), 
[Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img), 
[AudioLDM](https://github.com/haoheliu/AudioLDM), and
[Zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w).
We also partially draw inspirations from 
[PandaGPT](https://github.com/yxuansu/PandaGPT),  
[GILL](https://github.com/kohjingyu/gill/), 
[CoDi](https://codi-gen.github.io/),
[Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA),
[LLaVA](https://github.com/haotian-liu/LLaVA),
and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4).
Thanks for their wonderful works.




## License Notices
This repository is under [BSD 3-Clause License](LICENSE.txt).
NExT-GPT is a research project intended for non-commercial use only. 
One must NOT use the code of NExT-GPT for any illegal, harmful, violent, racist, or sexual purposes. 
One is strictly prohibited from engaging in any activity that will potentially violate these guidelines.
Any potential commercial use of this code should be approved by the authors.
