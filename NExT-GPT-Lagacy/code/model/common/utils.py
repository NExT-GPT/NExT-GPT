import random

import torch
from torch.nn.utils import rnn

import io
import json
import logging
import os
import pickle
import re
import shutil
import urllib
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse


def truncate_caption(caption: str) -> str:
    """Truncate captions at periods and newlines."""
    caption = caption.strip('\n')
    trunc_index = caption.find('\n') + 1
    if trunc_index <= 0:
        trunc_index = caption.find('.') + 1
    if trunc_index > 0:
        caption = caption[:trunc_index]
    return caption


def build_one_instance_for_pgpt4(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0:  # the first human turn
            assert role == 'human'
            text = '### Human: </Img> ' + turn['value'] + '\n### Assistant: '
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = 'Human: ' + turn['value'] + '\n### Assistant: '
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100] * len(one_input_id)
            elif role == 'gpt':
                text = turn['value'] + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids


def build_one_instance_for_cc3m(tokenizer, conversation):
    text_list = []
    input_ids, target_ids = [], []
    turn_num = len(conversation)
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0:  # the first human turn
            assert role == 'human'
            text = '### Human: ' + turn['value'] + '\n### Assistant: '
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = 'Human: ' + turn['value'] + '\n### Assistant: '
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100] * len(one_input_id)
            elif role == 'gpt':
                if 'image_name' in turn.keys():
                    img_tokens = ' '.join([f'[IMG{i}]' for i in range(8)])
                    text = turn['value'] + ' ' + img_tokens + '\n###'
                else:
                    text = turn['value'] + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
                # if 'image_name' in turn.keys():
                #     img_tokens = ' '.join([f'[IMG{i}]' for i in range(8)])
                #     img_input_ids = tokenizer(img_tokens, add_special_tokens=False).input_ids
                #     input_ids += img_input_ids
                #     target_ids += img_input_ids
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids


def build_one_instance_for_cc3m_1(tokenizer, conversation, num_img_tokens=8):
    text_list = []
    input_ids, target_ids = [], []
    turn_num = len(conversation)
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0:  # the first human turn
            assert role == 'human'
            text = turn['value'] + '\n### Assistant: '
            # text = turn['value']
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += one_input_id  # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = turn['value']
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            elif role == 'gpt':
                # if 'image_name' in turn.keys():
                #     img_tokens = ' '.join([f'[IMG{i}]' for i in range(num_img_tokens)])
                #     text = turn['value'] + img_tokens
                # else:
                #     text = turn['value']
                text = ' '.join([f'[IMG{i}]' for i in range(num_img_tokens)])
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
                # if 'image_name' in turn.keys():
                #     img_tokens = ' '.join([f'[IMG{i}]' for i in range(8)])
                #     img_input_ids = tokenizer(img_tokens, add_special_tokens=False).input_ids
                #     input_ids += img_input_ids
                #     target_ids += img_input_ids
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids


def build_one_instance_for_webvid(tokenizer, conversation, num_video_tokens=8):
    text_list = []
    input_ids, target_ids = [], []

    # text = '### Human: ' + conversation + '\n### Assistant: '
    # one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    # input_ids += one_input_id
    # target_ids += one_input_id  # do not perform loss regression on human prompt

    video_tokens = ' '.join([f'[VID{i}]' for i in range(num_video_tokens)])
    text = conversation + video_tokens
    text_list.append(text)
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += one_input_id
    assert len(input_ids) == len(target_ids)

    return text_list, input_ids, target_ids


def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len, dataset='cc3m',
                           num_img_tokens=8, num_video_tokens=8):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        if dataset == "pgpt4":
            _, one_input_ids, one_target_ids = build_one_instance_for_pgpt4(tokenizer, conversation)
        elif dataset == 'cc3m' or dataset == 'coco2017':
            _, one_input_ids, one_target_ids = build_one_instance_for_cc3m_1(tokenizer, conversation, num_img_tokens)
        elif dataset == 'webvid':
            _, one_input_ids, one_target_ids = build_one_instance_for_webvid(tokenizer, conversation, num_video_tokens)
        else:
            raise Exception("not support dataset name, it should be pgpt4 or cc3m")
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


def mask_token(inputs, tokenizer, mlm_probability, vocab_size=None, special_tokens_mask=None):
    """
    randomly mask some input tokens
    """
    indices_replaced = torch.bernoulli(torch.full(inputs.shape, mlm_probability)).bool()
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs


def build_one_instance_stage_1(tokenizer, captions, prompt=''):
    input_ids, target_ids = [], []
    texts = ''
    text = '</Img> ' + prompt + '\n### Assistant: '
    texts += text
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt

    text = captions + '\n###'
    texts += text
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += one_input_id
    return input_ids, target_ids


def process_batch_stage_1(tokenizer, batch_of_captions, max_tgt_len, prompt=''):
    batch_input_ids, batch_target_ids = [], []
    for caption in batch_of_captions:
        one_input_ids, one_target_ids = build_one_instance_stage_1(tokenizer, caption, prompt)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


def build_one_instance_stage_2(tokenizer, captions, num_signal_tokens=4, MODALITY='image'):
    input_ids, target_ids = [], []
    text = captions + '\n### Assistant: '
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt

    if MODALITY == 'image':
        signal_tokens = ' '.join([f'[IMG{i}]' for i in range(num_signal_tokens)])
    elif MODALITY == 'video':
        signal_tokens = ' '.join([f'[VID{i}]' for i in range(num_signal_tokens)])
    elif MODALITY == 'audio':
        signal_tokens = ' '.join([f'[AUD{i}]' for i in range(num_signal_tokens)])
    else:
        signal_tokens = ''

    text = captions + signal_tokens + '\n###'
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += one_input_id
    return input_ids, target_ids


def process_batch_stage_2(tokenizer, batch_of_captions, max_tgt_len, num_signal_tokens=4, MODALITY='image'):
    """
    :param mode: the target modality
    :param num_tokens: the number of generated signal tokens for generation
    """
    batch_input_ids, batch_target_ids = [], []
    # batch_caption_lists = []
    for captions in batch_of_captions:
        one_input_ids, one_target_ids = build_one_instance_stage_2(tokenizer, captions,
                                                                   num_signal_tokens=num_signal_tokens,
                                                                   MODALITY=MODALITY)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
        # batch_caption_lists.append(caption)
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


# def process_batch_stage_2(tokenizer, batch_of_captions, )


def build_one_instance_stage_3(tokenizer, conversation, img_tokens=4, vid_tokens=24, aud_tokens=8):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0:  # the first human turn
            assert role == 'human'
            if turn['input_modality'] != 'text':
                text = '</Img> ' + turn['value'] + '\n### Assistant: '
            else:
                text = turn['value'] + '\n### Assistant: '
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = 'Human: ' + turn['value'] + '\n### Assistant: '
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100] * len(one_input_id)
            elif role == 'gpt':
                if turn['output_modality'] == 'image':
                    signal_tokens = ' '.join([f'[IMG{i}]' for i in range(img_tokens)])
                elif turn['output_modality'] == 'video':
                    signal_tokens = ' '.join([f'[VID{i}]' for i in range(vid_tokens)])
                elif turn['output_modality'] == 'audio':
                    signal_tokens = ' '.join([f'[AUD{i}]' for i in range(aud_tokens)])
                else:
                    signal_tokens = ''
                caption = turn['caption']
                text = turn['value'] + signal_tokens + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids, caption


def process_batch_stage_3(tokenizer, batch_of_conversations, max_tgt_len, img_tokens=4, vid_tokens=24, aud_tokens=8):
    """
    :param mode: the target modality
    :param num_tokens: the number of generated signal tokens for generation
    """
    batch_input_ids, batch_target_ids = [], []
    # batch_caption_lists = []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids, caption = build_one_instance_stage_3(tokenizer, conversation,
                                                                               img_tokens=img_tokens,
                                                                               vid_tokens=vid_tokens,
                                                                               aud_tokens=aud_tokens)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
        # batch_caption_lists.append(caption)
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    # if is_mask_token:
    #     input_ids = mask_token(input_ids, tokenizer, 0.5)
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


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


def get_modality(path_list):
    _postfix = os.path.splitext(path_list[0])[-1]
    if _postfix == '.jpg':
        return 'image'
    elif _postfix == '.wav':
        return 'audio'
    elif _postfix == '.mp4':
        return 'video'
    else:
        raise NotImplementedError
