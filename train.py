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
import os
import sys
import logging
import pathlib
from typing import Dict, List, Union

import transformers
import tokenizers
import torch
import torch.nn as nn
from nextgpt.model import *
from nextgpt.dataset.base_dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
from nextgpt.dataset.concat_dataset import MyConcatDataset
from training_utils import ModelArguments, DataArguments, TrainingArguments
from nextgpt import conversation as conversation_lib
from nextgpt_trainer import NextGPTTrainer

import warnings
warnings.filterwarnings("ignore")


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')




def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_input_projector', 'mm_output_img_projector', 'mm_output_aud_projector', 'mm_output_vid_projector', 'multimodal_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_input_adapter_for_hf_trainer(trainer: transformers.Trainer,
                                           output_dir: str):
    # Only save Adapter
    keys_to_match = ['mm_input_projector', 'embed_tokens', 'embed_in']
    # if getattr(trainer.args, "use_im_start_end", False):
    #     keys_to_match.extend(['embed_tokens', 'embed_in'])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split('/')[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith('checkpoint-'):
            mm_projector_folder = os.path.join(parent_folder, "mm_input_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f'mm_input_projector.bin'))
    return


def safe_save_output_adapter_for_hf_trainer(trainer: transformers.Trainer,
                                            output_dir: str):
    # Only save Adapter
    keys_to_match = ['mm_output_img_projector', 'mm_output_vid_projector', 'mm_output_aud_projector', 'embed_tokens', 'embed_in']
    trainer.model.config.save_pretrained(output_dir)
    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    

    current_folder = output_dir.split('/')[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith('checkpoint-'):
            mm_projector_folder = os.path.join(parent_folder, "mm_output_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f'mm_output_projector.bin'))
    return


def save_adapter_for_hf_trainer(trainer: transformers.Trainer,
                                output_dir: str, 
                                keys_to_match: List[str], 
                                adapter_name: str):
    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    current_folder = output_dir.split('/')[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith('checkpoint-'):
            mm_projector_folder = os.path.join(parent_folder, "mm_output_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
        else:
            torch.save(weight_to_save, os.path.join(output_dir, adapter_name))
    return True
    

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    trainer.model.config.save_pretrained(output_dir)
    save_flag = False
    if getattr(trainer.args, 'tune_mm_input_adapter', False):
        keys_to_match = ['mm_input_projector', 'vision_resampler', 'embed_tokens', 'embed_in']
        save_flag = save_adapter_for_hf_trainer(trainer, output_dir, keys_to_match, 'mm_input_projector.bin')
    
    if any(getattr(trainer.args, f'tune_mm_output_{mod}_adapter', False) for mod in ['img', 'vid', 'aud']):
        keys_to_match = ['mm_output_img_projector', 'mm_output_vid_projector', 'mm_output_aud_projector', 'embed_tokens', 'embed_in']
        save_flag = save_adapter_for_hf_trainer(trainer, output_dir, keys_to_match, 'mm_output_projector.bin')

    if save_flag:
        return
    # trainer.tokenizer.save_pretrained(output_dir)
    print("save model !!!!!")
    if trainer.deepspeed:
        torch.cuda.synchronize()
        new_output_dir = output_dir+'/model'
        if not os.path.exists(new_output_dir):
            os.makedirs(new_output_dir)
        trainer.save_model(new_output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
    #                             data_path=data_args.data_path,
    #                             data_args=data_args)
    print("Loading datasets...")
    train_dataset = MyConcatDataset(dataset_name_list=data_args.dataset_name_list,
                                    tokenizer=tokenizer,
                                    data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    
    global local_rank
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!

    # Arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.multimodal_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = NextGPTLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False


    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
        print(f"Building conversation: {conversation_lib.default_conversation}")  # 
        print(f"Building conversation.sep_style: {conversation_lib.default_conversation.sep_style}")  # SeparatorStyle.TWO

    if model_args.multimodal_tower is not None:
        model.get_model().initialize_input_multimodal_modules(
           model_args=model_args,
           fsdp=training_args.fsdp
        )

        mulitmodal_tower = model.get_multimodal_tower()
        mulitmodal_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        print(mulitmodal_tower.image_processor)
        print(mulitmodal_tower.video_processor)
        print(mulitmodal_tower.audio_processor)
        data_args.image_processor = mulitmodal_tower.image_processor
        data_args.video_processor = mulitmodal_tower.video_processor
        data_args.audio_processor = mulitmodal_tower.audio_processor
        model.config.n_img_tokens = data_args.n_img_tokens = model_args.n_img_tokens
        model.config.n_vid_tokens = data_args.n_vid_tokens = model_args.n_vid_tokens
        model.config.n_aud_tokens = data_args.n_aud_tokens = model_args.n_aud_tokens
        data_args.is_multimodal = True

        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        
        model.config.mm_use_img_start_end = data_args.mm_use_img_start_end = model_args.mm_use_img_start_end
        model.config.mm_input_projector_lr = training_args.mm_input_projector_lr
        training_args.use_img_start_end = model_args.mm_use_img_start_end
        model.config.mm_use_img_patch_token = model_args.mm_use_img_patch_token

        model.config.mm_use_vid_start_end = data_args.mm_use_vid_start_end = model_args.mm_use_vid_start_end
        model.config.mm_use_vid_patch_token = model_args.mm_use_vid_patch_token
        training_args.use_vid_start_end = model_args.mm_use_vid_start_end
        
        model.config.mm_use_aud_start_end = data_args.mm_use_aud_start_end = model_args.mm_use_aud_start_end
        model.config.mm_use_aud_patch_token = model_args.mm_use_aud_patch_token
        training_args.use_aud_start_end = model_args.mm_use_aud_start_end

        model.config.mm_output_projector_lr = training_args.mm_output_projector_lr
    
    if model_args.image_decoder is not None and model_args.video_decoder is not None and model_args.audio_decoder is not None:

        model.config.layer_idx = model_args.layer_idx   # setting Layer index to extract signal feature from LLM hidden states
        # model.config.snr_loss = training_args.snr_loss
        # model_args.has_img_gen_loss = training_args.has_img_gen_loss
        # model_args.has_vid_gen_loss = training_args.has_vid_gen_loss
        # model_args.has_aud_gen_loss = training_args.has_aud_gen_loss

        model.get_model().initialize_output_multimodal_modules(
            model_args=model_args
        )

        image_decoder = model.get_image_decoder()
        image_decoder.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        video_decoder = model.get_video_decoder()
        video_decoder.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        audio_decoder = model.get_audio_decoder()
        audio_decoder.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        model.get_model().mm_output_img_projector.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        model.get_model().mm_output_vid_projector.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        model.get_model().mm_output_aud_projector.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    
    model.requires_grad_(False)

    # freeze/unfreeze the mm input adapters   
    model.config.tune_mm_input_adapter = training_args.tune_mm_input_adapter = model_args.tune_mm_input_adapter
    if model_args.tune_mm_input_adapter:
        for p in model.get_model().mm_input_projector.parameters():
            p.requires_grad = True
    model.config.freeze_mm_input_adapter = training_args.freeze_mm_input_adapter
    if training_args.freeze_mm_input_adapter:
        for p in model.get_model().mm_input_projector.parameters():
            p.requires_grad = False

    # freeze/unfreeze the mm output image adapters
    model.config.tune_mm_output_img_adapter = training_args.tune_mm_output_img_adapter = model_args.tune_mm_output_img_adapter
    if model_args.tune_mm_output_img_adapter:
        # model.requires_grad_(False)
        for p in model.get_model().mm_output_img_projector.parameters():
            p.requires_grad = True
    model.config.freeze_mm_output_img_adapter = training_args.freeze_mm_output_img_adapter
    if training_args.freeze_mm_output_img_adapter:
        for p in model.get_model().mm_output_img_projector.parameters():
            p.requires_grad = False

    # freeze/unfreeze the mm output video adapters
    model.config.tune_mm_output_vid_adapter = training_args.tune_mm_output_vid_adapter = model_args.tune_mm_output_vid_adapter
    if model_args.tune_mm_output_vid_adapter:
        # model.requires_grad_(False)
        for p in model.get_model().mm_output_vid_projector.parameters():
            p.requires_grad = True
    model.config.freeze_mm_output_vid_adapter = training_args.freeze_mm_output_vid_adapter
    if training_args.freeze_mm_output_vid_adapter:
        for p in model.get_model().mm_output_vid_projector.parameters():
            p.requires_grad = False

    # freeze/unfreeze the mm output audio adapters
    model.config.tune_mm_output_aud_adapter = training_args.tune_mm_output_aud_adapter = model_args.tune_mm_output_aud_adapter
    if model_args.tune_mm_output_aud_adapter:
        # model.requires_grad_(False)
        for p in model.get_model().mm_output_aud_projector.parameters():
            p.requires_grad = True
    model.config.freeze_mm_output_aud_adapter = training_args.freeze_mm_output_aud_adapter
    if training_args.freeze_mm_output_aud_adapter:
        for p in model.get_model().mm_output_aud_projector.parameters():
            p.requires_grad = False

    # # print the model parameters to check if the adapters are trainable
    # for n, p in model.get_model().mm_input_projector.named_parameters():
    #     print(n, ': ', p.requires_grad)
    # for n, p in model.get_model().mm_output_aud_projector.named_parameters():
    #     print(n, ': ', p.requires_grad)
    # for n, p in model.get_model().mm_output_vid_projector.named_parameters():
    #     print(n, ': ', p.requires_grad)
    # for n, p in model.get_model().mm_output_img_projector.named_parameters():
    #     print(n, ': ', p.requires_grad)

    # initialize_vision_tokenizer
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)      
    model.print_model_parameters()

    print('Testing tokenizer ... \n')
    print("Tokenizer model max length:", tokenizer.tokenize(' '.join([f"<image_{i:02d}>" for i in range(model_args.n_img_tokens)])))
    print("Tokenizer model max length:", tokenizer.tokenize('hello generate the tokens, '+' '.join([f"<image_{i:02d}>" for i in range(model_args.n_img_tokens)])))
    print("Tokenizer model max length:", tokenizer.tokenize(' '.join([f"<video_{i:02d}>" for i in range(model_args.n_vid_tokens)])))
    print("Tokenizer model max length:", tokenizer.tokenize(' '.join([f"<audio_{i:02d}>" for i in range(model_args.n_aud_tokens)])))

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    data_args.device = training_args.device
    data_args.version = model_args.version
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = NextGPTTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    # print("trainer.state: ", trainer.state)
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
        
        # torch.cuda.synchronize()
        # new_output_dir = training_args.output_dir+'/model'
        # if not os.path.exists(new_output_dir):
        #     os.makedirs(new_output_dir)
        
        # model.save_pretrained(new_output_dir)
        # tokenizer.save_pretrained(new_output_dir)
        # tokenizer.config.save_pretrained(new_output_dir)
        
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()