# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from nextgpt.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    VIDEO_TOKEN_INDEX,
    AUDIO_TOKEN_INDEX
)
from .multimodal_encoder.builder import build_multimodal_tower
from .multimodal_projector.builder import build_input_projector, build_output_projector
from .multimodal_decoder.builder import builder_decoder


__all__ = ["NextGPTMetaModel", "NextGPTMetaForCausalLM"]


class NextGPTMetaModel:
    def __init__(self, config):
        super(NextGPTMetaModel, self).__init__(config)

        print("Building and initing model ================")
        print("config: ", config)

        if hasattr(config, "multimodal_input_tower"):
            self.multimodal_tower = build_multimodal_tower(config, delay_load=True)
            self.mm_input_projector = build_input_projector(config)
        
        if hasattr(config, "image_decoder"):
            self.image_decoder = builder_decoder(config, "image")
            mm_output_img_projector_type = getattr(config, "mm_output_img_projector_type", "transformer")
            mm_output_img_features = getattr(config, "mm_output_img_features", 1024 )  # 768- v1.5, 1024- v2.0
            mm_output_img_num_query_token = getattr(config, "mm_output_img_num_query_token", 77)
            self.mm_output_img_projector = build_output_projector(config, mm_output_img_projector_type, mm_output_img_features, mm_output_img_num_query_token)
            
        if hasattr(config, "video_decoder"):
            self.video_decoder = builder_decoder(config, "video")
            mm_output_vid_projector_type = getattr(config, "mm_output_vid_projector_type", "transformer")
            mm_output_vid_features = getattr(config, "mm_output_vid_features", 1024)
            mm_output_vid_num_query_token = getattr(config, "mm_output_vid_num_query_token", 77)
            self.mm_output_vid_projector = build_output_projector(config, mm_output_vid_projector_type, mm_output_vid_features, mm_output_vid_num_query_token)
            
        if hasattr(config, "audio_decoder"):
            self.audio_decoder = builder_decoder(config, "audio")
            mm_output_aud_projector_type = getattr(config, "mm_output_aud_projector_type", "transformer")
            mm_output_aud_features = getattr(config, "mm_output_aud_features", 512)
            mm_output_aud_num_query_token = getattr(config, "mm_output_aud_num_query_token", 1)
            self.mm_output_aud_projector = build_output_projector(config, mm_output_aud_projector_type, mm_output_aud_features, mm_output_aud_num_query_token)         
        
    def get_multimodal_tower(self):
        multimodal_tower = getattr(self, "multimodal_tower", None)
        if type(multimodal_tower) is list:
            multimodal_tower = multimodal_tower[0]
        return multimodal_tower

    def get_input_projector(self):
        input_projector = getattr(self, "mm_input_projector", None)
        if type(input_projector) is list:
            input_projector = input_projector[0]
        return input_projector

    def get_image_decoder(self):
        image_decoder = getattr(self, "image_decoder", None)
        if type(image_decoder) is list:
            image_decoder = image_decoder[0]
        return image_decoder
    
    def get_image_text_encoder(self):
        return self.get_image_decoder().text_encoder

    def get_image_tokenizer(self):
        return self.get_image_decoder().tokenizer
    
    def get_image_vae(self):
        return self.get_image_decoder().vae
    
    def get_image_unet(self):
        return self.get_image_decoder().unet
    
    def get_image_noise_scheduler(self):
        return self.get_image_decoder().scheduler

    def get_video_decoder(self):
        video_decoder = getattr(self, "video_decoder", None)
        if type(video_decoder) is list:
            video_decoder = video_decoder[0]
        return video_decoder

    def get_video_text_encoder(self):
        return self.get_video_decoder().text_encoder
    
    def get_video_tokenizer(self):
        return self.get_video_decoder().tokenizer
    
    def get_video_vae(self):
        return self.get_video_decoder().vae
    
    def get_video_unet(self):
        return self.get_video_decoder().unet
    
    def get_video_noise_scheduler(self):
        return self.get_video_decoder().scheduler

    def get_audio_decoder(self):
        audio_decoder = getattr(self, "audio_decoder", None)
        if type(audio_decoder) is list:
            audio_decoder = audio_decoder[0]
        return audio_decoder

    def get_audio_text_encoder(self):
        return self.get_audio_decoder().text_encoder
    
    def get_audio_tokenizer(self):
        return self.get_audio_decoder().tokenizer
    
    def get_audio_vae(self):
        return self.get_audio_decoder().vae
    
    def get_audio_unet(self):
        return self.get_audio_decoder().unet
    
    def get_audio_vocoder(self):
        return self.get_audio_decoder().vocoder

    def get_audio_noise_scheduler(self):
        return self.get_audio_decoder().scheduler
    
    def get_output_image_projector(self):
        output_image_projector = getattr(self, "mm_output_img_projector", None)
        if type(output_image_projector) is list:
            output_image_projector = output_image_projector[0]
        return output_image_projector
    
    def get_output_video_projector(self):
        output_video_projector = getattr(self, "mm_output_vid_projector", None)
        if type(output_video_projector) is list:
            output_video_projector = output_video_projector[0]
        return output_video_projector
    
    def get_output_audio_projector(self):
        output_audio_projector = getattr(self, "mm_output_aud_projector", None)
        if type(output_audio_projector) is list:
            output_audio_projector = output_audio_projector[0]
        return output_audio_projector
        
    def initialize_input_multimodal_modules(self, model_args, fsdp=None):
        multimodal_tower = getattr(model_args, 'multimodal_input_tower', getattr(model_args, 'multimodal_tower', None))
        pretrain_mm_input_adapter = getattr(model_args, 'pretrain_mm_input_adapter', None)
        self.config.multimodal_input_tower = multimodal_tower

        if self.get_multimodal_tower() is None:
            multimodal_tower = build_multimodal_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.multimodal_tower = [multimodal_tower]
            else:
                self.multimodal_tower = multimodal_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                multimodal_tower = self.multimodal_tower[0]
            else:
                multimodal_tower = self.multimodal_tower
            multimodal_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_input_projector_type = getattr(model_args, "mm_input_projector_type", "linear")
        self.config.mm_hidden_size = multimodal_tower.hidden_size

        if getattr(self, "mm_input_projector", None) is None:
            self.mm_input_projector = build_input_projector(self.config)
        else:
            print("mm_input_projector already exists.")
            for p in self.mm_input_projector.parameters():
                p.requires_grad = True
        if pretrain_mm_input_adapter is not None:
            print("Loading pretrain_mm_input_adapter from : ", pretrain_mm_input_adapter)
            mm_projector_weights = torch.load(pretrain_mm_input_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_input_projector.load_state_dict(state_dict=get_w(mm_projector_weights, "mm_input_projector"))

    def initialize_output_multimodal_modules(self, model_args):
        image_decoder = model_args.image_decoder
        video_decoder = model_args.video_decoder
        audio_decoder = model_args.audio_decoder

        self.config.image_decoder = image_decoder
        self.config.video_decoder = video_decoder
        self.config.audio_decoder = audio_decoder

        self.config.mm_output_img_projector_type = getattr(model_args, "mm_output_img_projector_type", "transformer")
        self.config.mm_output_vid_projector_type = getattr(model_args, "mm_output_vid_projector_type", "transformer")
        self.config.mm_output_aud_projector_type = getattr(model_args, "mm_output_aud_projector_type", "transformer")
        
        self.config.mm_output_img_features = getattr(model_args, "mm_output_img_features", 1024)
        self.config.mm_output_img_num_query_token = getattr(model_args, "mm_output_img_num_query_token", 77)

        self.config.mm_output_vid_features = getattr(model_args, "mm_output_vid_features", 1024)
        self.config.mm_output_vid_num_query_token = getattr(model_args, "mm_output_vid_num_query_token", 77)

        self.config.mm_output_aud_features = getattr(model_args, "mm_output_aud_features", 512)
        self.config.mm_output_aud_num_query_token = getattr(model_args, "mm_output_aud_num_query_token", 1)

        pretrain_mm_output_img_adapter = getattr(model_args, "pretrain_mm_output_img_adapter", None)
        pretrain_mm_output_vid_adapter = getattr(model_args, "pretrain_mm_output_vid_adapter", None)
        pretrain_mm_output_aud_adapter = getattr(model_args, "pretrain_mm_output_aud_adapter", None)
        
        if self.get_image_decoder() is None:
            image_decoder = builder_decoder(model_args, "image")
            self.image_decoder = image_decoder 
        
        if self.get_video_decoder() is None:
            video_decoder = builder_decoder(model_args, "video")
            self.video_decoder = video_decoder
        
        if self.get_audio_decoder() is None:
            audio_decoder = builder_decoder(model_args, "audio")
            self.audio_decoder = audio_decoder

        # build image output projector
        if getattr(self, "mm_output_img_projector", None) is None:
            print("Building image output projector ... ")
            self.mm_output_img_projector = build_output_projector(self.config, self.config.mm_output_img_projector_type, self.config.mm_output_img_features, self.config.mm_output_img_num_query_token)
        else:
            print("mm_output_img_projector already exists.")
            for p in self.mm_output_img_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_output_img_adapter is not None:
            print("Loading pretrain_mm_output_img_adapter from : ", pretrain_mm_output_img_adapter)
            mm_output_img_projector_weights = torch.load(pretrain_mm_output_img_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_output_img_projector.load_state_dict(state_dict=get_w(mm_output_img_projector_weights, "mm_output_img_projector"))
        
        # build video output projector
        if getattr(self, "mm_output_vid_projector", None) is None:
            print("Building video output projector ... ")
            self.mm_output_vid_projector = build_output_projector(self.config, self.config.mm_output_vid_projector_type, self.config.mm_output_vid_features, self.config.mm_output_vid_num_query_token)
        else:
            print("mm_output_vid_projector already exists.")
            for p in self.mm_output_vid_projector.parameters():
                p.requires_grad = True
        if pretrain_mm_output_vid_adapter is not None:
            print("Loading pretrain_mm_output_vid_adapter from : ", pretrain_mm_output_vid_adapter)
            mm_output_vid_projector_weights = torch.load(pretrain_mm_output_vid_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_output_vid_projector.load_state_dict(state_dict=get_w(mm_output_vid_projector_weights, "mm_output_vid_projector"))

        # build audio output projector
        if getattr(self, "mm_output_aud_projector", None) is None:
            print("Building audio output projector ... ")
            self.mm_output_aud_projector = build_output_projector(self.config, self.config.mm_output_aud_projector_type, self.config.mm_output_aud_features, self.config.mm_output_aud_num_query_token)
        else:
            print("mm_output_aud_projector already exists.")
            for p in self.mm_output_aud_projector.parameters():
                p.requires_grad = True
        if pretrain_mm_output_aud_adapter is not None:
            print("Loading pretrain_mm_output_aud_adapter from : ", pretrain_mm_output_aud_adapter)
            mm_output_aud_projector_weights = torch.load(pretrain_mm_output_aud_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_output_aud_projector.load_state_dict(state_dict=get_w(mm_output_aud_projector_weights, "mm_output_aud_projector"))
        

class NextGPTMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_multimodal_tower(self):
        return self.get_model().get_multimodal_tower()
    
    def get_input_projector(self):
        return self.get_model().get_input_projector()
    
    def get_image_decoder(self):
        return self.get_model().get_image_decoder()

    def get_image_text_encoder(self):
        return self.get_model().get_image_text_encoder()
    
    def get_image_tokenizer(self):
        return self.get_model().get_image_tokenizer()
    
    def get_image_vae(self):
        return self.get_model().get_image_vae()
    
    def get_image_unet(self):
        return self.get_model().get_image_unet()
    
    def get_image_noise_scheduler(self):
        return self.get_model().get_image_noise_scheduler()
    
    def get_video_decoder(self):
        return self.get_model().get_video_decoder()

    def get_video_text_encoder(self):
        return self.get_model().get_video_text_encoder()
    
    def get_video_tokenizer(self):
        return self.get_model().get_video_tokenizer()
    
    def get_video_vae(self):
        return self.get_model().get_video_vae()
    
    def get_video_unet(self):
        return self.get_model().get_video_unet()
    
    def get_video_noise_scheduler(self):
        return self.get_model().get_video_noise_scheduler()
    
    def get_audio_decoder(self):
        return self.get_model().get_audio_decoder()

    def get_audio_text_encoder(self):
        return self.get_model().get_audio_text_encoder()
    
    def get_audio_tokenizer(self):
        return self.get_model().get_audio_tokenizer()
    
    def get_audio_vae(self):
        return self.get_model().get_audio_vae()
    
    def get_audio_unet(self):
        return self.get_model().get_audio_unet()
    
    def get_audio_vocoder(self):
        return self.get_model().get_audio_vocoder()
    
    def get_audio_noise_scheduler(self):
        return self.get_model().get_audio_noise_scheduler()

    def get_output_image_projector(self):
        return self.get_model().get_output_image_projector()

    def get_output_video_projector(self):
        return self.get_model().get_output_video_projector()

    def get_output_audio_projector(self):
        return self.get_model().get_output_audio_projector()

    def encode_images(self, images):
        image_features = self.get_model().get_multimodal_tower()(images, modality='image')
        image_features = self.get_model().mm_input_projector(image_features)
        return image_features

    def encode_videos(self, videos):
        video_features = self.get_model().get_multimodal_tower()(videos, modality='video')
        video_features = self.get_model().mm_input_projector(video_features)
        return video_features
    
    def encode_audios(self, audios):
        audio_features = self.get_model().get_multimodal_tower()(audios, modality='audio')
        audio_features = self.get_model().mm_input_projector(audio_features)
        return audio_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images=None,
        videos=None,
        audios=None,
    ):
        multimodal_tower = self.get_multimodal_tower()

        if multimodal_tower is None or (images is None and videos is None and audios is None) or input_ids.shape[1] == 1:
            return (input_ids, position_ids, attention_mask, past_key_values, None, labels)
        
        if images is not None:
            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [(x.unsqueeze(dim=0) if x.ndim == 3 else x) for x in images]
                concat_images = torch.concat([x for x in images], dim=0)
                image_features = self.encode_images(concat_images)
            else:
                image_features = self.encode_images(images)   # [b, 4096]
        else:
            # dummy image features
            image_features =  torch.zeros((input_ids.shape[0], 4096), dtype=multimodal_tower.dtype, device=input_ids.device)
        
        if videos is not None:
            if type(videos) is list or videos.ndim == 7:
                if type(videos) is list:
                    videos = [(x.unsqueeze(dim=0) if x.ndim == 5 else x) for x in videos]
                concat_videos = torch.concat([video for video in videos], dim=0)
                video_features = self.encode_videos(concat_videos)
            else:
                video_features = self.encode_videos(videos)
        else:
            # dummy video features
            video_features =  torch.zeros((input_ids.shape[0], 4096), dtype=multimodal_tower.dtype, device=input_ids.device)
        
        if audios is not None:
            if type(audios) is list or audios.ndim == 6:
                if type(audios) is list:
                    audios = [(x.unsqueeze(dim=0) if x.ndim == 4 else x) for x in audios]
                concat_audios = torch.concat([audio for audio in audios], dim=0)
                audio_features = self.encode_audios(concat_audios)
            else:
                audio_features = self.encode_audios(audios)
        else:
            # dummy audio features
            audio_features =  torch.zeros((input_ids.shape[0], 4096), dtype=multimodal_tower.dtype, device=input_ids.device)

        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        
        
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        cur_video_idx = 0
        cur_audio_idx = 0
        total_images = 0
        total_videos = 0
        total_audios = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_videos = (cur_input_ids == VIDEO_TOKEN_INDEX).sum()
            num_audios = (cur_input_ids == AUDIO_TOKEN_INDEX).sum()
            total_images += num_images
            total_videos += num_videos
            total_audios += num_audios
            if num_images == 0 and num_videos == 0 and num_audios == 0:
                # print("No image, video, audio tokens found in the input.")
                cur_image_features = image_features[0]
                # print("cur_image_features: ", cur_image_features[0:0])
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue
                
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() 
            video_token_indices = torch.where(cur_input_ids == VIDEO_TOKEN_INDEX)[0].tolist()
            audio_token_indices = torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0].tolist()
            _specfical_token_indices = image_token_indices + video_token_indices + audio_token_indices
            _specfical_token_indices.sort()
            _specfical_token_indices = [-1] + _specfical_token_indices + [cur_input_ids.shape[0]]
            
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(_specfical_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[_specfical_token_indices[i] + 1 : _specfical_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[_specfical_token_indices[i] + 1 : _specfical_token_indices[i + 1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.concat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i, _token_indices in zip(range(num_images + num_videos + num_audios + 1), _specfical_token_indices[1:]):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if _token_indices in image_token_indices:
                    cur_mm_features = image_features[cur_image_idx]
                    cur_mm_features = cur_mm_features.unsqueeze(0) if cur_mm_features.ndim == 1 else cur_mm_features
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_mm_features)
                    cur_new_labels.append(torch.full((cur_mm_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                elif _token_indices in video_token_indices:
                    cur_mm_features = video_features[cur_video_idx]
                    cur_mm_features = cur_mm_features.unsqueeze(0) if cur_mm_features.ndim == 1 else cur_mm_features
                    cur_video_idx += 1
                    cur_new_input_embeds.append(cur_mm_features)
                    cur_new_labels.append(torch.full((cur_mm_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                elif _token_indices in audio_token_indices:
                    cur_mm_features = audio_features[cur_audio_idx]
                    cur_mm_features = cur_mm_features.unsqueeze(0) if cur_mm_features.ndim == 1 else cur_mm_features
                    cur_audio_idx += 1
                    cur_new_input_embeds.append(cur_mm_features)
                    cur_new_labels.append(torch.full((cur_mm_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                else:
                    continue  
            
            cur_new_input_embeds = torch.concat(cur_new_input_embeds)
            cur_new_labels = torch.concat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.concat(
                        (
                            torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[(i), -cur_len:] = cur_new_labels
                    attention_mask[(i), -cur_len:] = True
                    position_ids[(i), -cur_len:] = torch.arange(start=0, end=cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.concat((
                            cur_new_embed,
                            torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                        ),
                        dim=0))
                if cur_len > 0:
                    new_labels_padded[(i), :cur_len] = cur_new_labels
                    attention_mask[(i), :cur_len] = True
                    position_ids[(i), :cur_len] = torch.arange(start=0, end=cur_len, dtype=position_ids.dtype, device=position_ids.device)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        if _position_ids is None:
            position_ids = None
        return (None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels)

    def initialize_vision_tokenizer(self, model_args, tokenizer):

        print("Initializing vision tokenizer ...")
        print("original vocab size: ", len(tokenizer))
        # add modality tokens
        signal_token_list = []
        signal_token_list.extend([f"<image_{i:02d}>" for i in range(model_args.n_img_tokens)])
        signal_token_list.extend([f"<video_{i:02d}>" for i in range(model_args.n_vid_tokens)])
        signal_token_list.extend([f"<audio_{i:02d}>" for i in range(model_args.n_aud_tokens)])

        num_new_tokens = tokenizer.add_tokens(signal_token_list, special_tokens=True)
        print(f"Adding {num_new_tokens} new tokens to the tokenizer.")
        self.resize_token_embeddings(len(tokenizer)) 
        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data
            print("self.get_input_embeddings().weight.data: ", self.get_input_embeddings().weight.requires_grad)
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
        if model_args.tune_mm_input_adapter or model_args.tune_mm_output_img_adapter or model_args.tune_mm_output_vid_adapter or model_args.tune_mm_output_aud_adapter:
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in self.get_output_embeddings().parameters():
                p.requires_grad = True

        if getattr(model_args, "pretrain_mm_input_adapter", None) is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_input_adapter)
            embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
            # assert num_new_tokens == 2
            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(
                    f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                )
        if  getattr(model_args, "pretrain_mm_output_img_adapter", None) is not None or getattr(model_args, "pretrain_mm_output_vid_adapter", None) is not None or getattr(model_args, "pretrain_mm_output_aud_adapter", None) is not None:
            output_adapter = None
            if getattr(model_args, "pretrain_mm_output_img_adapter", None) is not None:
                output_adapter = model_args.pretrain_mm_output_img_adapter
            elif getattr(model_args, "pretrain_mm_output_vid_adapter", None) is not None:
                output_adapter = model_args.pretrain_mm_output_vid_adapter
            elif getattr(model_args, "pretrain_mm_output_aud_adapter", None) is not None:
                output_adapter = model_args.pretrain_mm_output_aud_adapter
            # mm_projector_weights = torch.load(output_adapter)
            mm_projector_weights = torch.load(output_adapter)
            embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
            # assert num_new_tokens == 2

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(
                    f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                )
        
        if model_args.mm_use_img_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
        if model_args.mm_use_img_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
            if model_args.tune_mm_input_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.stop_gradient = not True
                for p in self.get_output_embeddings().parameters():
                    p.stop_gradient = not False
            if model_args.pretrain_mm_input_adapter:
                mm_projector_weights = torch.load(path=model_args.pretrain_mm_input_adapter)
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_img_patch_token:
            if model_args.tune_mm_input_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True
        
