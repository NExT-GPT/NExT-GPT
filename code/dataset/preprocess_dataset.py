import json
import os.path

from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import re
import random
import numpy as np
import torch


def load_alpaca(data_path, sample_data=False, sample_numer=1000, save_dir=''):
    """
    sample and process the alpaca dataset in to the following format:
    [
        {
            "image_name": "00000000000",
            "output_modality": "text",
            "conversation": [
                {
                    "from": "human",
                    "value": "Give three tips for staying healthy.",
                    "input_modality": "text"
                },
                {
                    "from": "gpt",
                    "value": "1. Eat a balanced and nutritious diet: ...",
                    "caption": "",
                    "output_modality": "text"
                }
            ]
        },
        ...
    ]
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    print('the total instance is {}'.format(len(data)))
    if sample_data and sample_numer > 0:
        data = random.sample(data, sample_numer)
    res = []
    for d in data:
        _temp = dict()
        _temp['image_name'] = '00000000000'
        _temp['output_modality'] = 'text'
        conversation = []

        conversation.append(
            {'from': 'human',
             'value': d['instruction'] + d['input'],
             'input_modality': 'text'}
        )
        conversation.append(
            {'from': 'gpt',
             'value': d['output'],
             'caption': '',
             'output_modality': 'text'}
        )
        _temp['conversation'] = conversation
        res.append(_temp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, os.path.basename(data_path))
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)
    return res


def load_llava(data_path, sample_data=False, sample_numer=1000, save_dir=''):
    """
    sample and process the llava instruction dataset into the following format:
    [
        {
            "image_name": "00000000000.jpg",
            "output_modality": "text",
            "conversation": [
                {
                    "from": "human",
                    "value": "Give three tips for staying healthy.",
                    "input_modality": "image"
                },
                {
                    "from": "gpt",
                    "value": "1. Eat a balanced and nutritious diet: ...",
                    "caption": "",
                    "output_modality": "text"
                }
            ]
        },
        ...
    ]
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    print('the total instance is {}'.format(len(data)))
    if sample_data and sample_numer > 0:
        res = random.sample(data, sample_numer)
    else:
        res = data
    # res = data
    save_path = os.path.join(save_dir, os.path.basename(data_path))
    for x in res:
        i = 0
        x['output_modality'] = 'text'
        for j in x['conversation']:
            if j['from'] == 'gpt':
                j['caption'] = ''
                j['output_modality'] = 'text'
            elif j['from'] == 'human':
                if i == 0:
                    j['input_modality'] = 'image'
                    i += 1
                else:
                    j['input_modality'] = 'text'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)
    return res


def load_t2x(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    save_dir = '../../data/IT_data/T+X-T_data'
    res = []

    # audios = load_t2x(os.path.join(save_dir, 'audio_t2x.json'))
    # videos = load_t2x(os.path.join(save_dir, 'video_t2x.json'))
    # images = load_t2x(os.path.join(save_dir, 'image_t2x.json'))
    # sample_number = max(len(audios), len(videos), len(images))
    #
    # print(sample_number)
    sample_number = 1000

    print('Load aplaca dataset ...')
    text = load_alpaca('../../data/IT_data/T+X-T_data/alpaca/alpaca.json', False, sample_number, save_dir)
    res.extend(text)

    print('Load llava dataset ...')
    data = load_llava('../../data/IT_data/T+X-T_data/llava/llava.json', False, sample_number, save_dir)
