import os
import json
import random

from tqdm import tqdm


def is_other_input(data):
    flag_output = 0
    flag_input = 0
    for d in data:
        if d.startswith('Human'):
            if d.count('<Image>') + d.count('<Audio>') + d.count('<Video>') > 0:
                flag_input = 1
        if d.startswith('GPT'):
            if d.count('<Image>') + d.count('<Audio>') + d.count('<Video>') > 0:
                flag_output = 1
    if flag_input == 0 and flag_output == 1:
        return True
    else:
        return False


def process_gpt_log():
    res = []
    with open('./demonstrations/gpt_log.txt', 'r', encoding='utf-8') as f:
        _temp = []
        for line in f.readlines():
            if line.startswith('---'):
                if len(_temp) > 0:  # there must be content in the current dialogue
                    if is_other_input(_temp):  # the input should not include the image/video/audio
                        if _temp[0].startswith('Human') and _temp[-1].startswith(
                                'GPT'):  # the dailogue should be start with Human, and end with GPT
                            _temp = ''.join(_temp)
                            if len(_temp.split()) < 300:  # the number of tokens in a dialogue should not too long.
                                if '<Image>' in _temp or '<Audio>' in _temp or '<Video>' in _temp:
                                    res.append(''.join(_temp))
                _temp = []
            elif line.startswith('Dialogue'):
                pass
            elif line == '\n':
                pass
            else:
                _temp.append(line)

    with open('./demonstrations/process_1.txt', 'w', encoding='utf-8') as f:
        for r in res:
            f.write(r)
            f.write('-------------------------\n')

# process_gpt_log()


def process_data_to_json():
    res = []
    with open('./demonstrations/process_1.txt', 'r', encoding='utf-8') as f:
        human, gpt = [], []
        conversation = []
        for line in f.readlines():
            if line.startswith('-------------------'):
                if len(gpt) > 0:
                    conversation.append({'from': 'gpt', 'value': ''.join(gpt).split('GPT: ')[-1]})
                res.append({'image_name': '00000.jpg', 'conversation': conversation})
                conversation = []
                human, gpt = [], []
            elif line.startswith('Human'):
                if len(gpt) > 0:
                    conversation.append({'from': 'gpt', 'value': ''.join(gpt).split('GPT: ')[-1]})
                    gpt = []
                human.append(line)
            elif line.startswith('GPT'):
                if len(human) > 0:
                    conversation.append({'from': 'human', 'value': ''.join(human).split('Human: ')[-1]})
                    human = []
                gpt.append(line)
            else:
                if len(human) > 0:
                    human.append(line)
                elif len(gpt) > 0:
                    gpt.append(line)
                else:
                    pass

    print(len(res))
    with open('./demonstrations/data.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)


process_gpt_log()
process_data_to_json()
