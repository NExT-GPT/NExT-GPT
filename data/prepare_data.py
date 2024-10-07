import json 
import random
import pandas as pd
from tqdm import tqdm
import os
import re


DEFINED_COMPREHENSION_PROMPTS = [
    "Give me a summary of the {}.",
    "What is the {} about?",
    "Tell me about the {}.",
    "Give me a brief summary of the {}.",
    "What is the main idea of the {}?",
    "Give me a short summary of the {}.",
    "Generate a caption about {}.",
    "Summarize the {}.",
    "What is the {}?",
    "What can you see from the {}?",
    "What is the main point of the {}?",
    "Give me a short description of the {}.",
    "What is the {} trying to say?",
    "Give me a brief description of the {}.",
    "What is the {} saying?",
    "What is the {} trying to convey?",
    "What is the {} trying to express?",
    "Can you outline the essence of the {}?",
    "Describe the core message of the {}.",
    "Provide an overview of the {}.",
    "What's the central theme of the {}?",
    "Could you encapsulate the essence of the {}?",
    "I'm curious about the essence of the {}.",
    "Could you distill the {} into a few sentences?",
    "Tell me the key points of the {}.",
    "What are the key elements of the {}?",
    "I'd like to know more about the {}.",
    "What's the primary focus of the {}?",
    "What's the {} really about?",
    "Give me an essence of the {}.",
    "What's the {} conveying?",
    "What's the central idea behind the {}?"
]



DEFINED_GENERATION_PROMPTS = {
    "Generate a {} about {}.",
    "Create a {} about {}.",
    "Show me a {} about {}.",
}

IMAGE_KEYWORDS = ['picture', 'image', 'scene', 'photograph', 'snapshot', 'illustration', 'vision', 'photo', 'drawing', 'painting', 'view', 'vista']
VIDEO_KEYWORDS = ['video', 'scene', 'film', 'clip', 'visual content', 'movie', 'motion picture', 'footage', 'cinematic work', 'video clip', 'video recording', 'video content']
AUDIO_KEYWORDS = ['sound', 'audio', 'recording', 'melody', 'voice', 'music', 'rhythm', 'tune', 'song', 'soundtrack', 'audio clip', 'audio recording', 'audio content']
INSTRUCTION_PROMPT = ['{} me a {} about ', '{} me a {} of ', '{} {} from ', '{} a {} where ', 'I\'d love to {} a {} of ', 'I\'d like to {} a {} of ', 'I\'d like you to {} a {} of ',
                      'please {} a {} of ', 'Can you {} me a {} of ', 'Could you {} me a {} of ', 'I want you to {} a {} of ', 'I need you to {} a {} of ', 
                      'I would like you to {} a {} of ']

PRODUCE_KEYWORDS = ['generate', 'show', 'synthesize', 'produce', 'create', 'yield', 'form', 'manufacture', 'fabricate',
                    'compose', 'originate', 'make', 'render', 'illustrate', 'demonstrate', 'exhibit', 'display', 'depict', 'present', 'visualize', 'design', 
                    'craft', 'develop', 'construct', 'build', 'forge', 'shape', 'mold', 'fashion', 'conjure']

RESPONSE_TEMPLATE = ['Sure, this is the {} you want.', 'Here is a {} for your reference', 'for reference, is this ok?',
                     'Ok.', 'No problem.', 'Sure.', 'how about this?',
                     'i would like to help.', 'this is what you want.', 'this is what you look for.', 'Certainly!',
                     'Sure thing!',  'of course', 'Absolutely!', 'Definitely', 'Certainly!', 'Sure thing!', 'Of course!']



def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def preprocess_generation_data(data, modality='image', save_path=None):
    """
    Preprocess the data for generation task, add a conversation between human and GPT
    """
    new_data = []
    for idx, d in tqdm(enumerate(data), total=len(data)):
        caption = d['caption']
        if modality == 'image':
            image_name = d['image_name']
            instruction = random.choice(INSTRUCTION_PROMPT).format(random.choice(PRODUCE_KEYWORDS), random.choice(IMAGE_KEYWORDS)) + caption
            new_data.append({
                "output_image": image_name,
                "image_captions": caption,
                "image_caption_embeddings": f'{image_name}.npy',
                "conversations": [
                    {
                        "from": "human",
                        "value": instruction
                    },
                    {
                        "from": "gpt",
                        "value": random.choice(RESPONSE_TEMPLATE).format(modality) + " " + "<image>"
                    }
                ]
            })
        elif modality == 'video':
            video_name = d['video_name']
            instruction = random.choice(INSTRUCTION_PROMPT).format(random.choice(PRODUCE_KEYWORDS), random.choice(VIDEO_KEYWORDS)) + caption
            new_data.append({
                "output_video": video_name,
                "video_captions": caption,
                "video_caption_embeddings": f'{video_name}.npy',
                "conversations": [
                    {
                        "from": "human",
                        "value": instruction
                    },
                    {
                        "from": "gpt",
                        "value": random.choice(RESPONSE_TEMPLATE).format(modality) + " " + "<video>"
                    }
                ]
            })
        elif modality == 'audio':
            audio_name = d['audio_name']
            instruction = random.choice(INSTRUCTION_PROMPT).format(random.choice(PRODUCE_KEYWORDS), random.choice(AUDIO_KEYWORDS)) + caption
            new_data.append({
                "output_audio": audio_name,
                "audio_captions": caption,
                "audio_caption_embeddings": f'{audio_name}.npy',
                "conversations": [
                    {
                        "from": "human",
                        "value": instruction
                    },
                    {
                        "from": "gpt",
                        "value": random.choice(RESPONSE_TEMPLATE).format(modality) + " " + "<audio>"
                    }
                ]
            })
        else:
            raise ValueError("Invalid modality")
    with open(save_path, 'w') as f:
        json.dump(new_data, f, indent=4)

    return new_data


def preprocess_comprehension_data(data, modality='image', save_path=None):
    """
    Preprocess the data for comprehension task, add a conversation between human and GPT
    """
    new_data = []
    for idx, d in tqdm(enumerate(data), total=len(data)):
        caption = d['caption']
        if modality == 'image':
            image_name = d['image_name']
            new_data.append({
                "input_image": image_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": '<image>\n' + random.choice(DEFINED_COMPREHENSION_PROMPTS).format(modality)
                    },
                    {
                        "from": "gpt",
                        "value": caption
                    }
                ]
            })
        elif modality == 'video':
            video_name = d['video_name']
            new_data.append({
                "input_video": video_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": '<video>\n' + random.choice(DEFINED_COMPREHENSION_PROMPTS).format(modality)
                    },
                    {
                        "from": "gpt",
                        "value": caption
                    }
                ]
            })
        elif modality == 'audio':
            audio_name = d['audio_name']
            new_data.append({
                "input_audio": audio_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": '<audio>\n' + random.choice(DEFINED_COMPREHENSION_PROMPTS).format(modality)
                    },
                    {
                        "from": "gpt",
                        "value": caption
                    }
                ]
            })
        else:
            raise ValueError("Invalid modality")
    
    with open(save_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    return data



def preprocess_cc3m(data_path):
    # load data
    data_json = []
    file_list = os.listdir(data_path)
    for file_name in tqdm(file_list, total=len(file_list)):
        if file_name.endswith('.json'):
            with open(os.path.join(data_path, file_name), 'r', encoding='utf-8') as f:
                _temp = json.load(f)
                if os.path.exists(os.path.join('./data/T_X_pair_data/cc3m/images', str(_temp['key'])+'.jpg')):
                    data_json.append({
                        "caption": _temp['caption'],
                        "id": _temp['key'],
                        "image_name": str(_temp['key'])+'.jpg',
                    })
    print(len(data_json))
    with open('./data/T_X_pair_data/cc3m/cc3m.json', 'w') as f:
        json.dump(data_json[:20], f, indent=4)
    return data_json


def preprocess_llava_pretrain(data_path):
    # prepara llava data for generation task

    with open(data_path, 'r') as f:
        _data = json.load(f)
    data = []
    for idx, d in tqdm(enumerate(_data), total=len(_data)):
        data.append({
            "id": d["id"],
            "image_name": d['image'],
            "caption": d["conversations"][1]["value"],
            "conversations": d["conversations"],
        })
    with open('./T_X_pair_data/llava_pretrain/llava_pretrain.json', 'w') as f:
        json.dump(data, f, indent=4)
    return data


def preprocess_webvid(data_path):
    """
    load video dataset into a json file
    """
    data_json = []
    # load error video-caption pairs
    error_list = []
    with open('./T_X_pair_data/webvid/error.txt', 'r') as f:
        for line in f:
            error_list.append(line.strip().split('/')[-1])
    print(len(error_list))
    error_list = set(error_list)
    print(len(error_list))
    # exit()
    file_list = os.listdir(data_path)
    for file_name in tqdm(file_list, total=len(file_list)):
        if file_name.endswith('.json'):
            with open(os.path.join(data_path, file_name), 'r', encoding='utf-8') as f:
                _temp = json.load(f)
                if os.path.exists(os.path.join('./data/T_X_pair_data/webvid/videos', str(_temp['key'])+'.mp4')):
                    if str(_temp['key'])+'.mp4' not in error_list:
                        data_json.append({
                            "caption": _temp['caption'],
                            "id": _temp['videoid'],
                            "video_name": str(_temp['key'])+'.mp4',
                        })
    print(len(data_json))
    with open('./T_X_pair_data/webvid/webvid.json', 'w') as f:
        json.dump(data_json, f, indent=4)
    return data_json


def preprocess_audiocap(data_path):
    # load audio dataset from 'audiocap' into a json file
    train_df = pd.read_csv(data_path, header=0)
    test_df = pd.read_csv(data_path.replace('train', 'test'), header=0)
    val_df = pd.read_csv(data_path.replace('train', 'val'), header=0)
    df = pd.concat([train_df, test_df, val_df], axis=0)
    data_json = []
    error_audio_list = [11251, 103491, 105063, 105067, 102977, 106789, 106641, 508, 107392, 104140, 
                        103586, 104593, 103479, 104600, 107259, 105723, 106016, 103468, 104639, 106267, 
                        107382, 103799, 107306, 102931, 102749, 103905, 104663, 105022, 17223, 105609]
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if os.path.exists(os.path.join('./data/T_X_pair_data/audiocap/audios', str(row["audiocap_id"])+".wav")):
            if row["audiocap_id"] in error_audio_list:
                continue
            audio_instance = {
                "audio_id": row["audiocap_id"],
                "audio_name": str(row["audiocap_id"])+".wav",
                "caption": row["caption"],
            }
            data_json.append(audio_instance)
    with open('./T_X_pair_data/audiocap/audiocap.json', 'w') as f:
        json.dump(data_json, f, indent=4)
    print(len(data_json))  #  44665
    return data_json


def preprocess_wavcaps(data_path):
    """
    load audio dataset into a json file
    """
    file_list = ["AudioSet_SL/as_final.json", "BBC_Sound_Effects/bbc_final.json", "FreeSound/fsd_final.json", "SoundBible/sb_final.json"]
    data = []
    for file_name in file_list:
        with open(os.path.join(data_path, file_name), 'r', encoding='utf-8') as f:
            _temp = json.load(f)
            for idx, d in enumerate(_temp['data']):
                pass
    return data


def preprocess_it_audio(data_path):
    """
    prepare audio data for instruction tuning
    """
    data = []
    with open(data_path, 'r') as f:
        _data = json.load(f)
    
    _temp = []
    for instance in _data:
        audio_captions = []
        audio_caption_embeddings = []
        conversation = instance['conversation']
        for conv in conversation:
            if "caption" in conv:
                audio_captions.append(conv['caption'])
            if conv['from'] == "gpt":
                conv['value'] = conv['value'] + " " + "<audio>"
        
        source_data = instance['source_data']
        audio_caption_embeddings.append(str(source_data["audiocap_id"])+".wav.npy")
        _temp.extend(audio_caption_embeddings)
        assert len(audio_captions) == len(audio_caption_embeddings)
        data.append({
            "audio_captions": audio_captions,
            "audio_caption_embeddings": audio_caption_embeddings,
            "conversations": conversation,
            "source_data": source_data
        })
    print(len(data))  # 9599
    print(len(set(_temp)))  # 4836
    with open(os.path.dirname(data_path)+'/audio_instruction_tuning.json', 'w') as f:
        json.dump(data, f, indent=4)


def preprocess_it_image(data_path):
    """
    prepare image data for instruction tuning
    """
    data = []
    with open(data_path, 'r') as f:
        _data = json.load(f)
    
    for idx, instance in enumerate(_data):
        image_captions = []
        image_caption_embeddings = []
        conversation = instance['conversation']
        for conv in conversation:
            if "caption" in conv:
                image_captions.append(conv['caption'])
            if conv['from'] == "gpt":
                conv['value'] = conv['value'] + " " + "<image>"
        
        source_data = instance['source_data']
        image_caption_embeddings.append("it_img_"+str(idx)+".jpg.npy")
        assert len(image_captions) == len(image_caption_embeddings)
        data.append({
            "image_captions": image_captions,
            "image_caption_embeddings": image_caption_embeddings,
            "conversations": conversation,
            "source_data": source_data
        })
    print(len(data))  # 9467
    with open(os.path.dirname(data_path)+'/image_instruction_tuning.json', 'w') as f:
        json.dump(data, f, indent=4)


def preprocess_it_video(data_path):
    """
    prepare video data for instruction tuning
    """
    data = []
    with open(data_path, 'r') as f:
        _data = json.load(f)
    
    for idx, instance in enumerate(_data):
        video_captions = []
        video_caption_embeddings = []
        conversation = instance['conversation']
        for conv in conversation:
            if "caption" in conv:
                video_captions.append(conv['caption'])
            if conv['from'] == "gpt":
                conv['value'] = conv['value'] + " " + "<video>"

        source_data = instance['source_data']
        video_caption_embeddings.append("it_vid_"+str(idx)+".mp4.npy")
        assert len(video_captions) == len(video_caption_embeddings)
        data.append({
            "video_captions": video_captions,
            "video_caption_embeddings": video_caption_embeddings,
            "conversations": conversation,
            "source_data": source_data
        })
    print(len(data))  # 9626
    with open(os.path.dirname(data_path)+'/video_instruction_tuning.json', 'w') as f:
        json.dump(data, f, indent=4)


def preprocess_it_mosit(data_path):
    """
    prepare mosit data for instruction tuning
    """
    data = []
    with open(data_path, 'r') as f:
        _data = json.load(f)
    
    for instance in _data:
        print(instance)
        conversations = instance['conversation']
        img_pattern = r'<Image>(.*?)</Image>'
        vid_pattern = r'<Video>(.*?)</Video>'
        aud_pattern = r'<Audio>(.*?)</Audio>'
        _temp_image_list = []
        _temp_video_list = []
        _temp_audio_list = []
        for conv in conversations:
            if conv['from'] == "gpt":
                matches = re.findall(r'<Image>(mosit_i_.*?\.jpg)</Image>', conv['value'])
                conv['value'] = re.sub(r'<image>(.*?)</image>', "", conv['value'])
                conv['value'] = re.sub(r'<Image>(mosit_i_.*?\.jpg)</Image>', "<image>", conv['value'])
                conv['value'] = re.sub(r'<Image>(.*?)</Image>', "", conv['value'])
                assert len(matches) ==  len(re.findall(r'<image>', conv['value']))
                if len(matches) > 0:
                    for m in matches:
                        _temp_image_list.append(m)
                matches = re.findall(r'<Video>(mosit_v_.*?\.mp4)</Video>', conv['value'])
                conv['value'] = re.sub(r'<video>(.*?)</video>', "", conv['value'])
                conv['value'] = re.sub(r'<Video>(mosit_v_.*?\.mp4)</Video>', "<video>", conv['value'])
                conv['value'] = re.sub(vid_pattern, "", conv['value'])
                assert len(matches) ==  len(re.findall(r'<video>', conv['value']))
                if len(matches) > 0:
                    for m in matches:
                        _temp_video_list.append(m)
                matches = re.findall(r'<Audio>(mosit_a_.*?\.wav)</Audio>', conv['value'])
                conv['value'] = re.sub(r'<audio>(.*?)</audio>', "", conv['value'])
                conv['value'] = re.sub(r'<Audio>(mosit_a_.*?\.wav)</Audio>', "<audio>", conv['value'])
                conv['value'] = re.sub(aud_pattern, "", conv['value'])
                assert len(matches) ==  len(re.findall(r'<audio>', conv['value']))
                if len(matches) > 0:
                    for m in matches:
                        _temp_audio_list.append(m)
                
            if conv['from'] == "human":
                conv['value'] = re.sub(img_pattern, "", conv['value'])
                conv['value'] = re.sub(vid_pattern, "", conv['value'])
                conv['value'] = re.sub(aud_pattern, " ", conv['value'])

        image_list = instance['image_list']
        image_captions = []
        image_caption_embeddings = []
        _new_instance = {}
        if len(image_list) > 0:
            for img in image_list:
                if img['image_name'] in _temp_image_list:
                    image_captions.append(img['caption'])
                    image_caption_embeddings.append(str(img['image_name'])+".npy")
            _new_instance["image_captions"] = image_captions
            _new_instance["image_caption_embeddings"] = image_caption_embeddings
        video_list = instance['video_list']
        video_captions = []
        video_caption_embeddings = []
        if len(video_list) > 0:
            for vid in video_list:
                if vid['video_name'] in _temp_video_list:
                    video_captions.append(vid['caption'])
                    video_caption_embeddings.append(str(vid['video_name'])+".npy")
            _new_instance["video_captions"] = video_captions
            _new_instance["video_caption_embeddings"] = video_caption_embeddings
        audio_list = instance['audio_list']
        audio_captions = []
        audio_caption_embeddings = []
        if len(audio_list) > 0:
            for aud in audio_list:
                if aud['audio_name'] in _temp_audio_list:
                    audio_captions.append(aud['caption'])
                    audio_caption_embeddings.append(str(aud['audio_name'])+".npy")
            _new_instance["audio_captions"] = audio_captions
            _new_instance["audio_caption_embeddings"] = audio_caption_embeddings
        

        _new_instance["conversations"] = conversations
        image_matches = re.findall(r'<image>', ' '.join([conv['value'] for conv in conversations]))
        assert len(image_matches) == len(image_captions), f"{len(image_matches)} != {len(image_captions)}\n{instance}"
        video_matches = re.findall(r'<video>', ' '.join([conv['value'] for conv in conversations]))
        assert len(video_matches) == len(video_captions), f"{len(video_matches)} != {len(video_captions)}\n{instance}"
        audio_matches = re.findall(r'<audio>', ' '.join([conv['value'] for conv in conversations]))
        assert len(audio_matches) == len(audio_captions), f"{len(audio_matches)} != {len(audio_captions)}\n{instance}"
        data.append(_new_instance)
    print(len(data))
    with open(os.path.dirname(data_path)+'/mosit_instruction_tuning.json', 'w') as f:
        json.dump(data, f, indent=4)
        

def preprocess_llava(data_path):
    with open(data_path, 'r') as f:
        _data = json.load(f)
    data = []
    for idx, d in tqdm(enumerate(_data), total=len(_data)):
        conversations = d["conversation"]
        for conv in conversations:
            if conv['from'] == "human":
                if random.random() > 0.5:
                    conv['value'] = conv['value'] + "\n<image>"
                else:
                    conv['value'] = '<image>\n' + conv['value']
        data.append({
            "input_image": [d["image_name"] for _ in range(len(re.findall(r'<image>', ' '.join([conv['value'] for conv in conversations]))))],
            "conversations": conversations,
        })
    with open(os.path.join(os.path.dirname(data_path), 'preprocessed_llava.json'), 'w') as f:
        json.dump(data, f, indent=4)


def preprocess_alpaca(data_path):
    
    with open(data_path, 'r') as f:
        _data = json.load(f)
    
    data = []
    for idx, d in tqdm(enumerate(_data), total=len(_data)):
        human_instruct = (d["instruction"] + " " + d["input"]).strip()
        gpt_answer = d["output"]
        conversation = [
            {
                "from": "human",
                "value": human_instruct
            },
            {
                "from": "gpt",
                "value": gpt_answer
            }
        ]
        data.append({
            "conversations": conversation,
        })
    with open(os.path.join(os.path.dirname(data_path), 'preprocessed_alpaca.json'), 'w') as f:
        json.dump(data, f, indent=4)




if __name__ == "__main__":

    # preprocess_webvid('./data/T_X_pair_data/webvid/results_2M_train.csv') 
    # preprocess_webvid('./data/T_X_pair_data/webvid/videos')  
    # preprocess_audiocap('./data/T_X_pair_data/audiocap/train.csv')  
    # preprocess_cc3m('./data/T_X_pair_data/cc3m/images') 
    
    # prepare data for comprehension and generation task
    # data = load_data('./data/T_X_pair_data/audiocap/audiocap.json')
    # new_data = preprocess_comprehension_data(data, modality='audio', save_path='./data/T_X_pair_data/audiocap//audiocap_comprehension.json')
    # new_data = preprocess_generation_data(data, modality='audio', save_path='./data/T_X_pair_data/audiocap/audiocap_generation.json')

    # data = load_data('./data/T_X_pair_data/cc3m/cc3m.json')
    # new_data = preprocess_comprehension_data(data, modality='image', save_path='./data/T_X_pair_data/cc3m/cc3m_comprehension.json')
    # new_data = preprocess_generation_data(data, modality='image', save_path='./data/T_X_pair_data/cc3m/cc3m_generation.json')

    # data = load_data('./data/T_X_pair_data/webvid/webvid.json')
    # new_data = preprocess_comprehension_data(data, modality='video', save_path='./data/T_X_pair_data/webvid/webvid_comprehension.json')
    # new_data = preprocess_generation_data(data, modality='video', save_path='./data/T_X_pair_data/webvid/webvid_generation.json')

    # data = preprocess_llava_pretrain('./data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json')
    # new_data = preprocess_comprehension_data(data, modality='image', save_path='./T_X_pair_data/llava_pretrain/llava_pretrain_comprehension.json')
    # new_data = preprocess_generation_data(data, modality='image', save_path='./T_X_pair_data/llava_pretrain/llava_pretrain_generation.json')


    # preprocess data for instruction tuning
    # preprocess_it_audio('./data/IT_data/T-T+X_data/audio_t2x.json')
    # preprocess_it_image('./data/IT_data/T-T+X_data/image_t2x.json')   
    # preprocess_it_video('./data/IT_data/T-T+X_data/video_t2x.json')
    # preprocess_it_mosit('./data/IT_data/MosIT_data/mosit.json')

    # preprocess_llava('./data/IT_data/T+X-T_data/llava-150k/llava.json')
    # preprocess_alpaca('./data/IT_data/T+X-T_data/alpaca/alpaca.json')

    pass