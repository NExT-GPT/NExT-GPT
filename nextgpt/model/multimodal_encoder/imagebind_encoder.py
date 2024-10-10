# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from .ImageBind import *
from .imagebind_processor import ImageProcessor, AudioProcessor, VideoProcessor
# from ImageBind import data
import torch
import torch.nn as nn
import os
from PIL import Image
from transformers.modeling_utils import get_parameter_dtype, get_parameter_device


class ImageBindTower(nn.Module):
    def __init__(self, multimodal_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.multimodal_tower_name = multimodal_tower
        # if not delay_load:
        self.load_model()
        # elif getattr(args, "unfreeze_mm_vision_tower", False):
        #     self.load_model()
        # else:
        #     raise ValueError("delay_load is True, but `unfreeze_mm_vision_tower` is False.")
        
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        self.image_processor = ImageProcessor()
    
    def load_model(self, device=None):
        if self.is_loaded:
            print("{} is already loaded, `load_model` called again, skipping.".format(self.multimodal_tower_name))
            return
        self.multimodal_tower, self.visual_hidden_size = imagebind_model.imagebind_huge(pretrained=True, store_path=self.multimodal_tower_name)
        # self.multimodal_tower.to('cuda')
        # TODO: check if this is necessary
        if device is not None:
            self.multimodal_tower.to(device)  

        for param in self.multimodal_tower.parameters():
            param.requires_grad = False
        self.is_loaded = True
    
    @torch.no_grad()
    def forward(self, inputs, modality='image'):
        """
        Args:
            inputs: List[Str], the path of input data.
            modality: ModalityType, the modality of input data.
        """
        _inputs = {}
        if modality == 'image':
            _inputs.update({ModalityType.VISION: inputs})  # [1, 3, 224, 224]
            _modality = ModalityType.VISION
        elif modality == 'audio':
            _inputs.update({ModalityType.AUDIO: inputs})  #  [1, 3, 1, 128, 204]
            _modality = ModalityType.AUDIO
        elif modality == 'video':
            _inputs.update({ModalityType.VISION: inputs})  # [1, 15, 3, 2, 224, 224]
            _modality = ModalityType.VISION
        else:
            raise ValueError("Modality not supported: {}".format(modality))
        
        # print({key: _inputs[key].shape for key in _inputs})
        # inputs = {key: _inputs[key].to(self.dtype) for key in _inputs}
        embeddings = self.multimodal_tower(_inputs)
        # print(embeddings[_modality].shape)
        return embeddings[_modality]
    
    @property
    def config(self):
        return self.multimodal_tower.config
    
    @property
    def hidden_size(self):
        return self.visual_hidden_size

    @property
    def device(self):
        return get_parameter_device(self)
    
    @property
    def dtype(self):
        # obtain the dtype of the model, which is the dtype of the first parameter
        return get_parameter_dtype(self)

if __name__ == "__main__":
    input_text = "A dog."
    input_image = "../../../assets/bird_image.jpg"
    input_audio = "../../../assets/test.wav"
    input_video = "../../../assets/test.mp4"
    model_name_or_path = "./pretrain_ckpt/imagebind"

    predictor = ImageBindTower(model_name_or_path, None, delay_load=False)
    processor = predictor.video_processor
    res = processor(input_video, return_tensors="pt")['pixel_values']
    res = predictor(res, modality='video')