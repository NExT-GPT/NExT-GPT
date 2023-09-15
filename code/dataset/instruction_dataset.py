import json
import os.path

from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import re
import random
import numpy as np
import torch
from .base_dataset import BaseDataset


INSTRUCTION_PROMPT = ['{} me a {} about ', '{} me a {} of ',
                      '{} {} from ', '{} a {} where ', 'I\'d love to {} a {} of '
                      'please {} a {} of ', 'Can you {} me a {} of ', 'Could you {} me a {} of ', ]
PRODUCE_KEYWORDS = ['generate', 'show', 'synthesize', 'produce', 'create', 'yield', 'form', 'manufacture', 'fabricate',
                    'compose', 'originate', 'make', 'render', 'illustrate', 'demonstrate']
IMAGE_KEYWORDS = ['picture', 'image', 'scene']
VIDEO_KEYWORDS = ['video', 'scene', 'film', 'clip', 'visual content', 'movie']
AUDIO_KEYWORDS = ['sound', 'audio', 'recording', 'melody', 'voice', 'music', 'rhythm']
RESPONSE_TEMPLATE = ['Sure, this is the {} you want.', 'Here is a {} for your reference', 'For reference, is this ok?',
                     'Ok.', 'No problem.', 'Sure.', 'How about this?',
                     'I would like to help.', 'This is what you want.', 'This is what you look for.', 'Certainly!',
                     'Sure thing!',  'Of course', 'Absolutely!', 'Definitely']


def load_audiocap(data_path, save_dir):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('the total instance is {}'.format(len(data)))
    res = []
    for row in tqdm(data, total=len(data)):
        audio_id, one_caption = row["audio_name"], row["caption"]
        _temp = {}
        instruction = INSTRUCTION_PROMPT[random.randint(0, 1)]
        produce_kw = PRODUCE_KEYWORDS[random.randint(0, len(PRODUCE_KEYWORDS) - 1)]
        audio_kw = AUDIO_KEYWORDS[random.randint(0, len(AUDIO_KEYWORDS) - 1)]

        _temp['image_name'] = audio_id
        _temp['output_modality'] = 'audio'
        conversation = []

        _temp_idx = random.randint(0, len(RESPONSE_TEMPLATE) - 1)
        response = RESPONSE_TEMPLATE[0].format(audio_kw) if _temp_idx == 0 else RESPONSE_TEMPLATE[_temp_idx]
        conversation.append(
            {'from': 'human',
             'value': instruction.format(produce_kw, audio_kw) + one_caption,
             'input_modality': 'text'}
        )
        conversation.append(
            {'from': 'gpt',
             'value': response,
             'caption': one_caption,
             'output_modality': 'audio'}
        )
        _temp['conversation'] = conversation
        res.append(_temp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, os.path.basename(data_path)), 'w', encoding='-utf-8') as f:
        json.dump(res, f, indent=4)
    return res, len(data)


def load_webvid(data_path, sample_number=1000, save_dir=''):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('the total instance is {}'.format(len(data)))
    data = random.sample(data, sample_number)
    res = []
    for row in tqdm(data, total=len(data)):
        video_name, one_caption = row["video_name"], row["caption"]
        _temp = dict()
        instruction = INSTRUCTION_PROMPT[random.randint(0, 1)]
        produce_kw = PRODUCE_KEYWORDS[random.randint(0, len(PRODUCE_KEYWORDS) - 1)]
        video_kw = VIDEO_KEYWORDS[random.randint(0, len(VIDEO_KEYWORDS) - 1)]
        _temp['image_name'] = video_name
        _temp['output_modality'] = 'video'
        conversation = []
        _temp_idx = random.randint(0, len(RESPONSE_TEMPLATE) - 1)
        response = RESPONSE_TEMPLATE[0].format(video_kw) if _temp_idx == 0 else RESPONSE_TEMPLATE[_temp_idx]
        conversation.append(
            {'from': 'human',
             'value': instruction.format(produce_kw, video_kw) + one_caption,
             'input_modality': 'text'}
        )
        conversation.append(
            {'from': 'gpt',
             'value': response,
             'caption': one_caption,
             'output_modality': 'video'}
        )
        _temp['conversation'] = conversation
        res.append(_temp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, os.path.basename(data_path)), 'w', encoding='-utf-8') as f:
        json.dump(res, f, indent=4)
    return res


def load_cc3m(data_path, sample_number=1000, save_dir=''):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('the total instance is {}'.format(len(data)))
    data = random.sample(data, sample_number)
    res = []
    for row in tqdm(data, total=len(data)):
        image_name, one_caption = row["image_name"], row["caption"]
        _temp = dict()
        instruction = INSTRUCTION_PROMPT[random.randint(0, 1)]
        produce_kw = PRODUCE_KEYWORDS[random.randint(0, len(PRODUCE_KEYWORDS) - 1)]
        image_kw = IMAGE_KEYWORDS[random.randint(0, len(IMAGE_KEYWORDS) - 1)]

        _temp['image_name'] = image_name
        _temp['output_modality'] = 'image'
        conversation = []
        _temp_idx = random.randint(0, len(RESPONSE_TEMPLATE) - 1)
        response = RESPONSE_TEMPLATE[0].format(image_kw) if _temp_idx == 0 else RESPONSE_TEMPLATE[_temp_idx]
        conversation.append(
            {'from': 'human',
             'value': instruction.format(produce_kw, image_kw) + one_caption,
             'input_modality': 'text'}
        )
        conversation.append(
            {'from': 'gpt',
             'value': response,
             'caption': one_caption,
             'output_modality': 'image'}
        )
        _temp['conversation'] = conversation
        res.append(_temp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, os.path.basename(data_path)), 'w', encoding='-utf-8') as f:
        json.dump(res, f, indent=4)
    return res


def load_alpaca(data_path, sample_numer=1000, save_dir=''):
    with open(data_path, 'r') as f:
        data = json.load(f)
    print('the total instance is {}'.format(len(data)))
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
        json.dump(data, f, indent=4)
    return res


def load_llava(data_path, sample_numer, save_dir=''):
    with open(data_path, 'r') as f:
        data = json.load(f)
    print('the total instance is {}'.format(len(data)))
    # res = random.sample(data, sample_numer)
    res = data
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


class InstructionDataset(BaseDataset):
    def __init__(self, data_path: str, image_root_path: str, embed_path: str):
        super(InstructionDataset, self).__init__(data_path, image_root_path, embed_path)

        self.embed_path = embed_path

        with open(data_path, 'r', encoding='utf-8') as f:
            res = json.load(f)

        print('Load modality-mixed output dataset ...')
        self.image_path_list, self.image_caption_list = [], []
        self.video_path_list, self.video_caption_list = [], []
        self.audio_path_list, self.audio_caption_list = [], []
        self.text_path_list, self.text_caption_list = [], []
        self.visual_QA_list = []
        for instance in tqdm(res, total=len(res)):
            # self.output_modality_list.append(instance['output_modality'])
            if instance['output_modality'] == 'image':
                self.image_caption_list.append(instance['conversation'])
                self.image_path_list.append(instance['image_name'])
            elif instance['output_modality'] == 'video':
                self.video_caption_list.append(instance['conversation'])
                self.video_path_list.append(instance['image_name'])
            elif instance['output_modality'] == 'audio':
                self.audio_caption_list.append(instance['conversation'])
                self.audio_path_list.append(instance['image_name'])
            else:
                self.text_caption_list.append(instance['conversation'])
                # self.text_path_list.append(instance['image_name'])
        # self.text_path_list = self.text_path_list[:len(self.image_path_list)]
        self.text_caption_list = random.sample(self.text_caption_list, len(self.image_path_list))  # self.text_caption_list[:len(self.image_path_list)]

        with open('../data/IT_data/T-T+X_data/llava_instruct_150k.json', 'r', encoding='utf-8') as f:
            _temp = json.load(f)
        for instance in tqdm(_temp, total=len(_temp)):
            self.visual_QA_list.append(instance['conversation'])
            self.text_path_list.append(image_root_path + instance['image_name'])
        sampled_data = random.sample(list(zip(self.visual_QA_list, self.text_path_list)), k=len(self.image_path_list))
        self.visual_QA_list, self.text_path_list = zip(*sampled_data)
        assert len(self.image_path_list) == len(self.video_path_list) == len(self.audio_path_list) == len(self.text_path_list)
        assert len(self.image_caption_list) == len(self.video_caption_list) == len(self.audio_caption_list) == len(self.text_caption_list) == len(self.visual_QA_list)
        assert len(self.image_path_list) == len(self.image_caption_list)
        print(f'[!] collect {len(self.image_path_list)} samples for training')

    def __len__(self):  # number of instances
        return len(self.image_path_list)

        # def __getitem__(self, i) -> Dict[str, torch.Tensor]: # how to get item, 取一个样本

    def __getitem__(self, i):

        with open(os.path.join(self.embed_path, str(self.video_path_list[i]) + '.npy'), 'rb') as f:
            video_clip_embs = torch.from_numpy(np.load(f, allow_pickle=True))  # (num_clip_tokens, 768)

        with open(os.path.join(self.embed_path, str(self.image_path_list[i]) + '.npy'), 'rb') as f:
            image_clip_embs = torch.from_numpy(np.load(f, allow_pickle=True))  # (num_clip_tokens, 768)

        with open(os.path.join(self.embed_path, str(self.audio_path_list[i]) + '.npy'), 'rb') as f:
            audio_clip_embs = torch.from_numpy(np.load(f, allow_pickle=True))  # (1, 512)

        return dict(text_path_list=self.text_path_list[i], output_texts=self.text_caption_list[i], visual_QA_list=self.visual_QA_list[i],
                    video_path_list=self.video_path_list[i], video_output_texts=self.video_caption_list[i], video_clip_embs=video_clip_embs,
                    audio_path_list=self.audio_path_list[i], audio_output_texts=self.audio_caption_list[i], audio_clip_embs=audio_clip_embs,
                    image_path_list=self.image_path_list[i], image_output_texts=self.image_caption_list[i], image_clip_embs=image_clip_embs
                    )

    def collate(self, instances):
        text_path_list, output_texts, visual_QA_list, video_path_list, video_output_texts, video_clip_embs,\
            audio_path_list, audio_output_texts, audio_clip_embs,\
            image_path_list, image_output_texts, image_clip_embs = tuple(
            [instance[key] for instance in instances] for key in ("text_path_list", "output_texts", "visual_QA_list",
                                                                  "video_path_list", "video_output_texts", "video_clip_embs",
                                                                  "audio_path_list", "audio_output_texts", "audio_clip_embs",
                                                                  "image_path_list", "image_output_texts", "image_clip_embs"))
        return dict(
            text_path_list=text_path_list, output_texts=output_texts, visual_QA_list=visual_QA_list,
            video_path_list=video_path_list, video_output_texts=video_output_texts,
            video_clip_embs=video_clip_embs,
            audio_path_list=audio_path_list, audio_output_texts=audio_output_texts,
            audio_clip_embs=audio_clip_embs,
            image_path_list=image_path_list, image_output_texts=image_output_texts,
            image_clip_embs=image_clip_embs
        )


if __name__ == '__main__':
    save_dir = '../../data/IT_data/T-T+X_data/'
    res = []
    print('Load AudioCap dataset ...')
    audio, sample_number = load_audiocap('../../data/T-X_pair_data/audiocap/audiocap.json', save_dir)
    res.extend(audio)

    print('Load CC3M dataset ...')
    image = load_cc3m('../../data/T-X_pair_data/cc3m/cc3m.json', sample_number, save_dir)
    res.extend(image)

    print('Load Webvid dataset ...')
    video = load_webvid('../../data/T-X_pair_data/webvid/webvid.json', sample_number, save_dir)
    res.extend(video)

    print('Load aplaca dataset ...')
    text = load_alpaca('../../data/IT_data/aplaca/alpaca_data_cleaned.json', sample_number, save_dir)
    res.extend(text)

    print('Load llava dataset ...')
    data = load_llava('../../data/IT_data/llava/llava_instruct_150k.json', sample_number, save_dir)
    # res.extend(data)

    print('The total instance is {}\n'.format(len(res)))
    random.shuffle(res)
    with open(os.path.join(save_dir, 'instruction_data.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)

