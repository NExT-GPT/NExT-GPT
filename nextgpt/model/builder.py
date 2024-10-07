#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from nextgpt.model import *
from nextgpt.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": "cuda", **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'nextgpt' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from nextgpt.model.language_model.nextgpt_llama import NextGPTConfig
            base_cfg_pretrained = NextGPTConfig.from_pretrained(model_base)
            print('Loading NExT-GPT from base model...')
            model = NextGPTLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=base_cfg_pretrained, ignore_mismatched_sizes=True, **kwargs)
            
            print('Initialize NExT-GPT...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
           
            print('Loading additional NExT-GPT weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading NExT-GPT from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            from nextgpt.model.language_model.nextgpt_llama import NextGPTConfig
            cfg_pretrained = NextGPTConfig.from_pretrained(model_base)
            print('cfg_pretrained: ', cfg_pretrained)
            model = NextGPTLlamaForCausalLM.from_pretrained(model_base, config=cfg_pretrained).to(device="cuda", dtype=torch.float16) 
            print("kwargs: ", kwargs)
            

            print('mm_input_projector device', model.get_model().mm_input_projector.device)
            print('mm_input_projector dtype', model.get_model().mm_input_projector.dtype)
            print('mm_output_img_projector device', model.get_model().mm_output_img_projector.device)
            print('mm_output_img_projector dtype', model.get_model().mm_output_img_projector.dtype)
            print('Model device...', model.get_model().device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = NextGPTLlamaForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                **kwargs
            )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    # print("model.config ", model.config)
    if 'nextgpt' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", False)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        # num_new_tokens = tokenizer.add_tokens(signal_token_list, special_tokens=True)
        if mm_use_im_patch_token or mm_use_im_start_end:
            model.resize_token_embeddings(len(tokenizer))

        multimodal_tower = model.get_multimodal_tower()
        multimodal_tower.to(device=model.device)
        print("multimodal_tower device: ", multimodal_tower.device)
        print("multimodal_tower dtype: ", multimodal_tower.dtype)
        mm_input_projector = model.get_input_projector().to(device=model.device)
        mm_output_img_projector = model.get_output_image_projector().to(device=model.device)
        mm_output_vid_projector = model.get_output_video_projector().to(device=model.device)
        mm_output_aud_projector = model.get_output_audio_projector().to(device=model.device)
        image_decoder = model.get_image_decoder().to(device=model.device)
        video_decoder = model.get_video_decoder().to(device=model.device)
        audio_decoder = model.get_audio_decoder().to(device=model.device)
        
        image_processor = multimodal_tower.image_processor
        video_processor = multimodal_tower.video_processor
        audio_processor = multimodal_tower.audio_processor


    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, video_processor, audio_processor, context_len, model.config