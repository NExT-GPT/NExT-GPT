# Copyright (c) 2024 torchtorch Authors. All Rights Reserved.
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

from dataclasses import dataclass
import warnings
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, GenerationConfig

from transformers.modeling_outputs import ModelOutput
from transformers.generation.utils import GenerateOutput

from ..nextgpt_arch import NextGPTMetaForCausalLM, NextGPTMetaModel
from transformers import StoppingCriteria, StoppingCriteriaList
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        text_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        imgage_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            caption alignemnt mse loss and image generation loss.
        video_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            caption alignemnt mse loss and video generation loss.
        audio_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            caption alignemnt mse loss and audio generation loss.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_loss: Optional[torch.FloatTensor] = None
    image_cap_loss: Optional[torch.FloatTensor] = None
    image_gen_loss: Optional[torch.FloatTensor] = None
    video_cap_loss: Optional[torch.FloatTensor] = None
    video_gen_loss: Optional[torch.FloatTensor] = None
    audio_cap_loss: Optional[torch.FloatTensor] = None
    audio_gen_loss: Optional[torch.FloatTensor] = None
   

# __all__ = [
#     "NextGPTLlamaModel",
#     "NextGPTLlamaForCausalLM",
# ]

def l2_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Args:
        u: (N, T_I_V_A.txt, D) tensor.
        v: (N, T_I_V_A.txt, D) tensor.
    Returns:
        l1_loss: (N,) tensor of summed L1 loss.
    """
    assert u.shape == v.shape, (u.shape, v.shape)
    return ((u - v) ** 2).sum(dim=-1) ** 0.5


class NextGPTConfig(LlamaConfig):
    model_type = "nextgpt_llama"


class NextGPTLlamaModel(NextGPTMetaModel, LlamaModel):
    config_class = NextGPTConfig

    def __init__(self, config: LlamaConfig):
        super(NextGPTLlamaModel, self).__init__(config)


class NextGPTLlamaForCausalLM(LlamaForCausalLM, NextGPTMetaForCausalLM):
    config_class = NextGPTConfig
    # base_model_prefix = "nextgpt"

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = NextGPTLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _get_output(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.Tensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = True,
            output_hidden_states: Optional[bool] = True,
            cache_position: Optional[torch.Tensor] = None,
            images: Union[Optional[torch.Tensor], Optional[List[torch.Tensor]]] = None,
            videos: Union[Optional[torch.Tensor], Optional[List[torch.Tensor]]] = None,
            audios: Union[Optional[torch.Tensor], Optional[List[torch.Tensor]]] = None,
            return_dict: Optional[bool] = None,
            ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, videos, audios
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,
        images: Union[Optional[torch.Tensor], Optional[List[torch.Tensor]]] = None,
        videos: Union[Optional[torch.Tensor], Optional[List[torch.Tensor]]] = None,
        audios: Union[Optional[torch.Tensor], Optional[List[torch.Tensor]]] = None,
        image_caption_embeddings: Optional[List[torch.Tensor]] = None,
        image_captions: Optional[List[str]] = None,
        image_signal_token_indices: Optional[List[int]] = None,
        target_images: Optional[torch.Tensor] = None,
        target_images_feature: Optional[torch.Tensor] = None,
        video_caption_embeddings: Optional[List[torch.Tensor]] = None,
        video_captions: Optional[List[str]] = None,
        video_signal_token_indices: Optional[List[int]] = None,
        target_videos: Optional[torch.Tensor] = None,
        target_videos_feature: Optional[torch.Tensor] = None,
        audio_caption_embeddings: Optional[List[torch.Tensor]] = None,
        audio_captions: Optional[List[str]] = None,
        audio_signal_token_indices: Optional[List[int]] = None,
        target_audios: Optional[torch.Tensor] = None,
        target_audios_feature: Optional[torch.Tensor] = None,
        # has_img_gen_loss: Optional[bool] = None,
        # has_vid_gen_loss: Optional[bool] = None,
        # has_aud_gen_loss: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        outputs = self._get_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            cache_position=cache_position,
            images=images,
            videos=videos,
            audios=audios,
            return_dict=return_dict,
        )
        text_loss = outputs.loss
        img_cap_loss = None
        img_loss = None
        vid_cap_loss = None
        vid_loss = None
        aud_cap_loss = None
        aud_loss = None
        loss = text_loss
        
        has_img_gen_loss = True if target_images is not None or target_images_feature is not None else False
        _loss, img_cap_loss, img_loss = self.compute_image_loss(image_caption_embeddings, image_captions, 
                                                                target_images=target_images, has_gen_loss=has_img_gen_loss, has_snr_loss=True,
                                                                labels=labels, image_signal_token_indices=image_signal_token_indices,
                                                                hidden_states=outputs.hidden_states)
        if _loss is not None:
            loss = loss + _loss

        has_vid_gen_loss = True if target_videos is not None or target_videos_feature is not None else False
        _loss, vid_cap_loss, vid_loss = self.compute_video_loss(video_caption_embeddings, video_captions, 
                                                                target_videos=target_videos, has_gen_loss=has_vid_gen_loss, has_snr_loss=False,
                                                                labels=labels, video_signal_token_indices=video_signal_token_indices,
                                                                hidden_states=outputs.hidden_states)
        if _loss is not None:
            loss = loss + _loss 
        
        has_aud_gen_loss = True if target_audios is not None or target_audios_feature is not None else False
        _loss, aud_cap_loss, aud_loss = self.compute_audio_loss(audio_caption_embeddings, audio_captions, 
                                                                target_audios=target_audios, has_gen_loss=has_aud_gen_loss, has_snr_loss=False,
                                                                labels=labels, audio_signal_token_indices=audio_signal_token_indices,
                                                                hidden_states=outputs.hidden_states)
        if _loss is not None:
            loss = loss + _loss 
        
        if not return_dict:
            output = (outputs.logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            text_loss=text_loss,
            image_cap_loss=img_cap_loss,
            image_gen_loss=img_loss,
            video_cap_loss=vid_cap_loss,
            video_gen_loss=vid_loss,
            audio_cap_loss=aud_cap_loss,
            audio_gen_loss=aud_loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_snr(self, timesteps, noise_scheduler):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr
    
    def compute_image_loss(self, 
                        #    projected_embeddings: torch.Tensor = None, 
                           image_caption_embeddings: List[torch.Tensor] =  None, 
                           image_captions: Optional[List[str]] = None,
                           target_images: Optional[torch.Tensor] = None,
                           output_image_feature: Optional[torch.Tensor] = None,
                           has_gen_loss: Optional[bool] = True,
                           has_snr_loss: Optional[bool] = False,
                           labels: Optional[torch.Tensor] = None,
                           image_signal_token_indices: Optional[List[int]] = None,
                           hidden_states: Optional[torch.Tensor] = None,
                           ):
        loss = None
        cap_loss = None
        gen_loss = None
        
        if image_caption_embeddings is not None or image_captions is not None:
            # for image-side alignment learning
            start_pos = (labels == image_signal_token_indices[0][1]).nonzero(as_tuple=False).tolist()
            end_pos = (labels == image_signal_token_indices[-1][1]).nonzero(as_tuple=False).tolist()
            assert 0 < len(start_pos) == len(end_pos) and len(end_pos) > 0, (start_pos, end_pos)
            hidden_embedding = []
            input_embedding = []
            for s, e in zip(start_pos, end_pos):
                assert e[0] == s[0], (s, e)
                assert e[1] - s[1] + 1 == len(image_signal_token_indices)*2-1, (s, e)
                hidden_embedding.append(hidden_states[self.config.layer_idx][s[0], s[1]:e[1] + 1, :])
                input_embedding.append(self.get_input_embeddings()(labels[s[0], s[1]:e[1] + 1]))
            hidden_embedding = torch.stack(hidden_embedding, dim=0)
            input_embedding = torch.stack(input_embedding, dim=0)
            projected_embeddings = self.get_output_image_projector()(hidden_embedding, input_embedding)  # (N, seq_len, 2048)
            if image_caption_embeddings is None:
                text_inputs = self.get_image_tokenizer()(
                    image_captions,
                    padding="max_length",
                    max_length=length,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=add_special_tokens
                ).to(self.device)
                image_caption_embeddings = self.get_image_text_encoder()(**text_inputs)[0]
            else:
                image_caption_embeddings = torch.stack(image_caption_embeddings, dim=0).to(self.device)
            cap_loss = l2_loss(projected_embeddings, image_caption_embeddings).mean()
        loss = cap_loss
        if not has_gen_loss:
            return loss, cap_loss, gen_loss
        else:
            # assert output_image is not None or output_image_feature is not None
            if target_images is not None or output_image_feature is not None:
                if output_image_feature is not None:
                    latents = DiagonalGaussianDistribution(output_image_feature).sample()
                else:
                    if len(target_images.shape) == 3:
                        target_images = target_images.unsqueeze(0)

                    latents = self.get_image_vae().encode(target_images).latent_dist.sample()
                latents = latents * self.get_image_vae().config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.get_image_noise_scheduler().config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = self.get_image_noise_scheduler().add_noise(latents, noise, timesteps)

                target = noise

                model_pred = self.get_image_unet()(noisy_latents, timesteps, projected_embeddings).sample

                if has_snr_loss:
                    snr = self.compute_snr(timesteps, self.get_image_noise_scheduler())
                    mse_loss_weights = (
                        torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    gen_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    gen_loss = gen_loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    gen_loss = gen_loss.mean()
                    
                else:
                    gen_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss += gen_loss
        return loss, cap_loss, gen_loss

    def compute_video_loss(self, 
                        #    projected_embeddings: torch.Tensor = None, 
                           video_caption_embeddings: List[torch.Tensor] =  None, 
                           video_captions: Optional[List[str]] = None,
                           target_videos: Optional[torch.Tensor] = None,
                           output_video_feature: Optional[torch.Tensor] = None,
                           has_gen_loss: Optional[bool] = True,
                           has_snr_loss: Optional[bool] = False,
                           labels: Optional[torch.Tensor] = None,
                           video_signal_token_indices: Optional[List[int]] = None,
                           hidden_states: Optional[torch.Tensor] = None,
                           ):
        loss = None
        cap_loss = None
        gen_loss = None

        if video_caption_embeddings is not None or video_captions is not None:
            start_pos = (labels == video_signal_token_indices[0][1]).nonzero(as_tuple=False).tolist()
            end_pos = (labels == video_signal_token_indices[-1][1]).nonzero(as_tuple=False).tolist()
            assert 0 < len(start_pos) == len(end_pos) and len(end_pos) > 0, (start_pos, end_pos)
            hidden_embedding = []
            input_embedding = []
            for (s, e) in zip(start_pos, end_pos):
                assert e[0] == s[0], (s, e)
                assert e[1] - s[1] + 1 == len(video_signal_token_indices)*2-1, (s, e)
                hidden_embedding.append(hidden_states[self.config.layer_idx][s[0], s[1]:e[1] + 1, :])
                input_embedding.append(self.get_input_embeddings()(labels[s[0], s[1]:e[1] + 1]))
            hidden_embedding = torch.stack(hidden_embedding, dim=0)
            input_embedding = torch.stack(input_embedding, dim=0)
            projected_embeddings = self.get_output_video_projector()(hidden_embedding, input_embedding)
            if video_caption_embeddings is None:
                text_inputs = self.get_video_tokenizer()(
                    video_captions,
                    padding="max_length",
                    max_length=length,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=add_special_tokens
                ).to(self.device)
                video_caption_embeddings = self.get_video_text_encoder()(**text_inputs)[0]
            else:
                video_caption_embeddings = torch.stack(video_caption_embeddings, dim=0).to(self.device)
            cap_loss = l2_loss(projected_embeddings, video_caption_embeddings).mean()
        loss = cap_loss

        if not has_gen_loss:
            return loss, cap_loss, gen_loss
        else:
            if target_videos is not None or output_video_feature is not None:
                if output_video_feature is not None:
                    latents = DiagonalGaussianDistribution(output_video_feature).sample()
                else:
                    if len(target_videos.shape) == 4:
                        target_videos = target_videos.unsqueeze(0)
                    batch_size, channels, num_frames, height, width = target_videos.shape
                    target_videos = target_videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
                    latents = self.get_video_vae().encode(target_videos).latent_dist.sample()
                latents = latents * self.get_video_vae().config.scaling_factor
                _, channels, height, width = latents.shape
                latents = latents.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.get_video_noise_scheduler().config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = self.get_video_noise_scheduler().add_noise(latents, noise, timesteps)
                target = noise
                model_pred = self.get_video_unet()(noisy_latents, timesteps, projected_embeddings).sample
                if has_snr_loss:
                    snr = self.compute_snr(timesteps, self.get_video_noise_scheduler())
                    mse_loss_weights = (
                        torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    gen_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    gen_loss = gen_loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    gen_loss = gen_loss.mean()
                    
                else:
                    gen_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss += gen_loss
        return loss, cap_loss, gen_loss
    
    def compute_audio_loss(self, 
                        #    projected_embeddings: torch.Tensor = None, 
                           audio_caption_embeddings: List[torch.Tensor] =  None, 
                           audio_captions: Optional[List[str]] = None,
                           target_audios: Optional[torch.Tensor] = None,
                           output_audio_feature: Optional[torch.Tensor] = None,
                           has_gen_loss: Optional[bool] = True,
                           has_snr_loss: Optional[bool] = False,
                           labels: Optional[torch.Tensor] = None,
                           audio_signal_token_indices: Optional[List[int]] = None,
                           hidden_states: Optional[torch.Tensor] = None,
                           ):
        loss = None
        cap_loss = None
        gen_loss = None
        if audio_caption_embeddings is not None or audio_captions is not None:
            start_pos = (labels == audio_signal_token_indices[0][1]).nonzero(as_tuple=False).tolist()
            end_pos = (labels == audio_signal_token_indices[-1][1]).nonzero(as_tuple=False).tolist()
            assert 0 < len(start_pos) == len(end_pos) and len(end_pos) > 0, (start_pos, end_pos)
            hidden_embedding = []
            input_embedding = []
            for s, e in zip(start_pos, end_pos):
                assert e[0] == s[0], (s, e)
                assert e[1] - s[1] + 1 == len(audio_signal_token_indices)*2-1, (s, e)
                hidden_embedding.append(hidden_states[self.config.layer_idx][s[0], s[1]:e[1] + 1, :])
                input_embedding.append(self.get_input_embeddings()(labels[s[0], s[1]:e[1] + 1]))
            hidden_embedding = torch.stack(hidden_embedding, dim=0)
            input_embedding = torch.stack(input_embedding, dim=0)
            projected_embeddings = self.get_output_audio_projector()(hidden_embedding, input_embedding)
            if audio_caption_embeddings is None:
                text_inputs = self.get_audio_tokenizer()(
                    audio_captions,
                    padding="max_length",
                    max_length=length,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=add_special_tokens
                ).to(self.device)
                audio_caption_embeddings = self.get_audio_text_encoder()(**text_inputs)[0]
            else:
                audio_caption_embeddings = torch.stack(audio_caption_embeddings, dim=0).to(self.device)
            audio_caption_embeddings = audio_caption_embeddings.view(audio_caption_embeddings.size(0), 1,
                                                            audio_caption_embeddings.size(1))
            cap_loss = l2_loss(projected_embeddings, audio_caption_embeddings).mean()
        loss = cap_loss

        if not has_gen_loss:
            return loss, cap_loss, gen_loss
        else:
            if target_audios is not None or output_audio_feature is not None:
                if output_audio_feature is not None:
                    latents = DiagonalGaussianDistribution(output_audio_feature).sample()
                else:
                    # print('target_audios: ', target_audios.size())  # (b, 1024, 64)
                    if len(target_audios.shape) == 3:
                        target_audios = target_audios.unsqueeze(1)
                    # print('target_audios: ', target_audios.size())  # (b, 1, 1024, 64)
                    latents = self.get_audio_vae().encode(target_audios).latent_dist.sample()
                # print('latents: ', latents.size())  # torch.Size([b, 8, 260, 16])
                latents = latents * self.get_audio_vae().config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.get_audio_noise_scheduler().config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = self.get_audio_noise_scheduler().add_noise(latents, noise, timesteps)

                target = noise

                model_pred = self.get_audio_unet()(noisy_latents, timesteps, class_labels=projected_embeddings.squeeze(1), encoder_hidden_states=None).sample

                if has_snr_loss:
                    snr = self.compute_snr(timesteps, self.get_audio_noise_scheduler())
                    mse_loss_weights = (
                        torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    gen_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    gen_loss = gen_loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    gen_loss = gen_loss.mean()
                    
                else:
                    gen_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss += gen_loss
        return loss, cap_loss, gen_loss
    

    @torch.no_grad()
    def _get_generation(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 200,
        top_p: Optional[float] = 10.0,
        temperature: Optional[float] = 0.1,
        stopping_criteria: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs):
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or videos is not None or audios is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, None, None, images, videos, audios
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)

            batch_size, seq_length = attention_mask.shape
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).expand((batch_size, seq_length))
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs = super().generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            # do_sample=True,
            # use_cache=True,
            stopping_criteria=stopping_criteria,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=True,
            output_attentions=True,
            **kwargs
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        # image_generation_config
        image_signal_token_indices: Optional[List[int]] = None,
        guidance_scale_for_img: Optional[float] = 7.5,
        num_inference_steps_for_img: Optional[int] = 50,
        # video_generation_config
        video_signal_token_indices: Optional[List[int]] = None,
        guidance_scale_for_vid: Optional[float] = 7.5,
        num_inference_steps_for_vid: Optional[int] = 50,
        height: Optional[int] = 320,
        width: Optional[int] = 576,
        num_frames: Optional[int] = 16,
        # audio_generation_config
        audio_signal_token_indices: Optional[List[int]] = None,
        guidance_scale_for_aud: Optional[float] = 7.5,
        num_inference_steps_for_aud: Optional[int] = 50,
        audio_length_in_s: Optional[float] = 5.0,
        # image_sizes: Optional[torch.Tensor] = None,
        stopping_criteria: Optional[Callable] = None,
        max_num_imgs: Optional[int] = 5,
        max_num_vids: Optional[int] = 5,
        max_num_auds: Optional[int] = 5,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        print("kwargs: ", kwargs)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        output_attentions = kwargs.pop("output_attentions", True)
        output_hidden_states = kwargs.pop("output_hidden_states", True)
        max_new_tokens = kwargs.pop("max_new_tokens", 200)
        top_p = kwargs.pop("top_p", 10.0)
        temperature = kwargs.pop("temperature", 0.1)
        # stopping_criteria = kwargs.pop("stopping_criteria", None)

        outputs = self._get_generation(
            input_ids=input_ids,
            images=images,
            videos=videos,
            audios=audios,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            stopping_criteria=stopping_criteria,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )

        generated_ids = outputs.sequences
        # the output hidden states is a tuple - 
        # one is the input hidden states of all layers (32 + 1(embedding layers)) 
        # and the other is the output hidden states of all layers (32 + 1(embedding layers))
        layer_idx = getattr(self.config, "layer_idx", -1)
        hidden_embedding = [x[layer_idx] for x in outputs.hidden_states[1:]]
        print('hidden_embedding: ', len(hidden_embedding))
        print('hidden_embedding: ', hidden_embedding[0].size())
        hidden_embedding = torch.cat(hidden_embedding, dim=1)

        return_outputs = {
            "sequences": generated_ids,
        }
        print('generated_ids: ', generated_ids)
        # Find up to max_num_rets [IMG] tokens, and their corresponding scores.
        all_gen_img_idx = [i for i, x in enumerate(generated_ids[0, :] == image_signal_token_indices[0][1]) if x][
                          :max_num_imgs]
        print('all_gen_img_idx: ', all_gen_img_idx)

        # Find up to max_num_rest [VID] tokens, and their corresponding scores.
        all_gen_vid_idx = [i for i, x in enumerate(generated_ids[0, :] == video_signal_token_indices[0][1]) if x][
                          :max_num_vids]
        print('all_gen_vid_idx: ', all_gen_vid_idx)

        # Find up to max_num_rest [AUD] tokens, and their corresponding scores.
        all_gen_aud_idx = [i for i, x in enumerate(generated_ids[0, :] == audio_signal_token_indices[0][1]) if x][
                          :max_num_auds]
        print('all_gen_aud_idx: ', all_gen_aud_idx)

        if len(all_gen_img_idx) == 0 and len(all_gen_vid_idx) == 0 and len(all_gen_aud_idx) == 0:
            # No [IMG], [VID], [AUD] tokens.
            # caption = self.llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # # return_outputs.append(truncate_caption(caption))
            # return_outputs.append(caption)
            return return_outputs
        else:
            if len(all_gen_img_idx) > 0:
                img_outputs = self.generate_images(generated_ids, hidden_embedding, all_gen_img_idx,
                                                   image_signal_token_indices, guidance_scale=guidance_scale_for_img,
                                                   num_inference_steps=num_inference_steps_for_img)
                # return_outputs.append({'vid': img_outputs})
                return_outputs['images'] = img_outputs
            if len(all_gen_vid_idx) > 0:
                vid_outputs = self.generate_videos(generated_ids, hidden_embedding, all_gen_vid_idx,
                                                   video_signal_token_indices, guidance_scale=guidance_scale_for_vid,
                                                   num_inference_steps=num_inference_steps_for_vid, height=height,
                                                   width=width, num_frames=num_frames)
                # return_outputs.append({'vid': vid_outputs})
                return_outputs['videos'] = vid_outputs
            if len(all_gen_aud_idx) > 0:
                aud_outputs = self.generate_audios(generated_ids, hidden_embedding, all_gen_aud_idx,
                                                   audio_signal_token_indices, guidance_scale=guidance_scale_for_aud,
                                                   num_inference_steps=num_inference_steps_for_aud,
                                                   audio_length_in_s=audio_length_in_s)
                # return_outputs.append({'aud': aud_outputs})
                return_outputs['audios'] = aud_outputs
        return return_outputs  

    def generate_images(self, generated_ids, embeddings, all_gen_idx, 
                        image_sigal_token_indices, 
                        guidance_scale=7.5, num_inference_steps=40):
        """
        To generate the images based on the embeddings
        generated_ids: the  index of the generated tokens
        embedding: the embeddings for synthesizing images
        all_gen_idx: the index of [IMG0] in the generated_ids
        """
        last_ret_idx = 0
        return_outputs = []
        generation_model = self.get_image_decoder().to(embeddings.dtype)
        n_img_tokens = len(image_sigal_token_indices)
        _image_sigal_token_indices = [pair[1] for pair in image_sigal_token_indices]
        for gen_idx in all_gen_idx:
            end_idx = gen_idx + n_img_tokens*2-1
            gen_list = generated_ids[0, gen_idx:end_idx:2].cpu().detach().numpy().tolist()
            print('gen_list: ', gen_list)
            assert gen_list == _image_sigal_token_indices, (generated_ids, _image_sigal_token_indices)
            print('embeddings: ', embeddings.size())
            print('gen_idx: ', gen_idx)
            print('end_idx: ', end_idx)
            raw_emb = embeddings[:, gen_idx:end_idx, :]  # (1, 8, 4096)
            print('raw_emb: ', raw_emb.size())

            # Produce generation embedding.
            gen_prefix = generated_ids[0, gen_idx:end_idx]
            gen_prefix_embs = self.get_input_embeddings()(gen_prefix)  # (1, T_I_V_A.txt, D)
            gen_emb = self.get_output_image_projector()(raw_emb, gen_prefix_embs)  # (1, 77, 768)

            if gen_emb.shape[1] != 77:
                bs = gen_emb.shape[0]
                clip_emb = 768
                gen_emb = gen_emb.reshape(bs, -1, clip_emb)  # (bs, T_I_V_A.txt, 768)
                seq_len = gen_emb.shape[1]
                gen_emb = torch.cat([gen_emb, 
                                      torch.to_tensor(torch.zeros((bs, 77 - seq_len, clip_emb), dtype=gen_emb.dtype), 
                                                                place=gen_emb.place)], 
                                    dim=1)

            image_outputs = generation_model(prompt_embeds=gen_emb,
                                             guidance_scale=guidance_scale,
                                             num_inference_steps=num_inference_steps).images

            
            return_outputs.append(generated_ids[:, last_ret_idx:end_idx])
            last_ret_idx = gen_idx + 1
            # return_outputs.append(truncate_caption(caption) + f' {gen_prefix}')
            return_outputs.append(image_outputs)
        return return_outputs

    def generate_videos(self, generated_ids, embeddings, all_gen_idx,
                        video_signal_token_indices,
                        guidance_scale=7.5, num_inference_steps=40, height=320, width=576, num_frames=16):
        """
        To generate videos based on the embeddings
        generated_ids: the  index of the generated tokens
        embedding: the embeddings for synthesizing videos
        all_gen_idx: the index of [VID0] in the generated_ids
        """
        return_outputs = []
        last_ret_idx = 0
        generation_model = self.get_video_decoder().to(embeddings.dtype)
        _video_signal_token_indices = [pair[1] for pair in video_signal_token_indices]
        n_vid_tokens = len(video_signal_token_indices)
        for gen_idx in all_gen_idx:
            end_idx = gen_idx + n_vid_tokens*2-1
            gen_list = generated_ids[0, gen_idx:end_idx:2].cpu().detach().numpy().tolist()
            print('gen_list: ', gen_list)
            assert gen_list == _video_signal_token_indices, (generated_ids, _video_signal_token_indices)
            raw_emb = embeddings[:, gen_idx:end_idx, :]  # (1, 8, 4096)
            
            gen_prefix = generated_ids[0, gen_idx:end_idx]
            gen_prefix_embs = self.get_input_embeddings()(gen_prefix)  # (1, T_I_V_A.txt, D)
            gen_emb = self.get_output_video_projector()(raw_emb, gen_prefix_embs)  # (1, 77, 768)

            if gen_emb.shape[1] != 77:
                print(f"Padding {gen_emb.shape} with zeros")
                bs = gen_emb.shape[0]
                clip_emb = 768
                gen_emb = gen_emb.reshape(bs, -1, clip_emb)  # (bs, T_I_V_A.txt, 768)
                seq_len = gen_emb.shape[1]
                gen_emb = torch.cat([gen_emb, 
                                      torch.to_tensor(torch.zeros((bs, 77 - seq_len, clip_emb), dtype=gen_emb.dtype), 
                                                                place=gen_emb.place)], 
                                    dim=1)
                print('Padded to', gen_emb.shape)

            video_outputs = generation_model(prompt_embeds=gen_emb,
                                             guidance_scale=guidance_scale,
                                             num_inference_steps=num_inference_steps, height=height,
                                             width=width, num_frames=num_frames).frames
            return_outputs.append(generated_ids[:, last_ret_idx:end_idx])
            last_ret_idx = gen_idx + 1
            # return_outputs.append(truncate_caption(caption) + f' {gen_prefix}')
            return_outputs.append(video_outputs)
        return return_outputs

    def generate_audios(self, generated_ids, embeddings, all_gen_idx,
                        audio_signal_token_indices,
                        guidance_scale=7.5, num_inference_steps=40, audio_length_in_s=5.0):
        """
        To generate videos based on the embeddings
        generated_ids: the  index of the generated tokens
        embedding: the embeddings for synthesizing audios
        all_gen_idx: the index of [AUD0] in the generated_ids
        """
        return_outputs = []
        last_ret_idx = 0
        generation_model = self.get_audio_decoder().to(embeddings.dtype)
        _audio_signal_token_indices = [pair[1] for pair in audio_signal_token_indices]
        n_aud_tokens = len(audio_signal_token_indices)
        for gen_idx in all_gen_idx:
            end_idx = gen_idx + n_aud_tokens*2-1
            gen_list = generated_ids[0, gen_idx:end_idx:2].cpu().detach().numpy().tolist()
            print('gen_list: ', gen_list)
            assert gen_list == _audio_signal_token_indices, (generated_ids, _audio_signal_token_indices)
            raw_emb = embeddings[:, gen_idx:end_idx, :]  # (1, 8, 4096)
            
            gen_prefix = generated_ids[0, gen_idx:end_idx]
            gen_prefix_embs = self.get_input_embeddings()(gen_prefix)  # (1, T_I_V_A.txt, D)
            gen_emb = self.get_output_audio_projector()(raw_emb, gen_prefix_embs)  # (1, 77, 768)
            print('gen_emb size:', gen_emb.size())
            bs = gen_emb.shape[0]
            hid_emb_size = gen_emb.shape[2]
            gen_emb = gen_emb.view(bs, hid_emb_size)

            audio_outputs = generation_model(prompt_embeds=gen_emb,
                                             guidance_scale=guidance_scale,
                                             num_inference_steps=num_inference_steps,
                                             audio_length_in_s=audio_length_in_s).audios[0]

            return_outputs.append(generated_ids[:, last_ret_idx:end_idx])
            last_ret_idx = gen_idx + 1
            # return_outputs.append(truncate_caption(caption) + f' {gen_prefix}')
            return_outputs.append(audio_outputs)
        return return_outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        videos = kwargs.pop("videos", None)
        audios = kwargs.pop("audios", None)
        # image_sizes = kwargs.pop("image_sizes", None)

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )

        if images is not None:
            inputs["images"] = images
        if videos is not None:
            inputs["videos"] = videos
        if audios is not None:
            inputs["audios"] = audios
        # if image_sizes is not None:
        #     inputs["image_sizes"] = image_sizes
        return inputs

    def print_model_parameters(self, use_4bit=False):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        lora = 0
        image = 0
        video = 0
        audio = 0
        linear = 0
        llama = 0
        imagebind = 0
        for name, param in self.model.named_parameters():
            # print(f"{name}: {param.numel():,d} :: {param.requires_grad}")
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            if 'lora' in name:
                lora += num_params
            elif 'mm_output_img_projector' in name:
                image += num_params
            elif 'mm_output_vid_projector' in name:
                video += num_params
            elif 'mm_output_aud_projector' in name:
                audio += num_params
            elif 'mm_input_projector' in name:
                linear += num_params
            elif name.startswith("layers") or name.startswith("embed_tokens") or name.startswith("norm.weight"):
                llama += num_params
            elif 'multimodal_tower' in name:
                imagebind += num_params
            else:
                pass

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        if use_4bit:
            trainable_params /= 2
        print(
            f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
        )
        print(f'lora params: {lora:,d} || video params: {video:,d} || audio params: {audio:,d} || image params: {image:,d}')
        print(f'linear params: {linear:,d} || imagebind params: {imagebind:,d} || llama params: {llama:,d}')


AutoConfig.register("nextgpt_llama", NextGPTConfig)
AutoModelForCausalLM.register(NextGPTConfig, NextGPTLlamaForCausalLM)

