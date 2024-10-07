import json
import os


exp_name = './exp0_2023_09_27_03_10_22/'

gpt_log_path = f'{exp_name}gpt_log_audio.txt'
caption_path = f'{exp_name}log.txt'

modality = 'audio'
postfix = '.wav'
save_path = f'{exp_name}{modality}_t2x.json'

with open(caption_path, 'r', encoding='utf-8') as f:
    captions = []
    for line in f.readlines():
        captions.append(line.split('\t')[-1].strip())


with open(gpt_log_path, 'r', encoding='utf-8') as f:
    dialogues = []
    _temp = []
    _instances = []
    # need to split the data by *****
    for line in f.readlines():
        if line.startswith('***********'):
            _instances.append(_temp)
            _temp = []
        else:
            if line != '\n':
                _temp.append(line.strip())
    assert len(captions) == len(_instances)
    for idx, (instance, cap) in enumerate(zip(_instances, captions)):
        if len(instance) > 11:
            pass
        else:
            _human = ''
            _gpt = ''
            idy = 0
            for ins in instance:
                if ins.startswith('Human:'):
                    if _human == '':
                        _human = ins.split('Human: ')[-1]
                elif ins.startswith('GPT:'):
                    if _gpt == '':
                        _gpt = ins.split('GPT: ')[-1]
                elif ins.startswith('----'):
                    if _human != '' and _gpt != '':
                        _res = [
                            {'from': 'human', 'value': _human, 'input_modality': 'text'},
                            {'from': 'gpt', 'value': _gpt, 'caption': cap, 'output_modality': modality}
                        ]
                        res = {'image_name': f't2x00{idx}{idy}'+postfix, 'output_modality': modality, 'conversation': _res}
                        dialogues.append(res)
                        _human = ''
                        _gpt = ''
                        idy += 1
                else:
                    continue
            if _human != '' and _gpt != '':
                _res = [
                    {'from': 'human', 'value': _human, 'input_modality': 'text'},
                    {'from': 'gpt', 'value': _gpt, 'caption': cap, 'output_modality': modality}
                ]
                res = {'image_name': f't2x00{idx}{idy}' + postfix, 'output_modality': modality, 'conversation': _res}
                dialogues.append(res)
    print(len(dialogues))

with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(dialogues, f, indent=4)



