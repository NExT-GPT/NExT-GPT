import random
from datetime import datetime
import backoff
import openai
import argparse
from functools import lru_cache
from transformers import logging

logging.set_verbosity_error()
from .utils import *


BEFORE_EXAMPLE = 'Now you are a scenarist, and use your imagination to create a dialog between a **Human** role and' \
                 ' a **GPT** role. Please DO remember, the dialogue should closely coincide with or revolve around ' \
                 'the semantic meaning of the provided scene description.\n\n --------------------\nHere are some ' \
                 'examples:\n'

AFTER_EXAMPLE = '--------------------\nDO remember, the generation of dialog should meet following rules:\n1. There is' \
                ' only one turn of dialogue, i.e., the **GPT** respond to the query by the **Human**.\n2. Diversify the' \
                ' form and style of dialogue content. \n3. Do not ask similar questions in different utterances, and ' \
                'also do NOT entirely copy the provided demonstration.\n--------------------\nPlease follow above ' \
                'instructions, and produce TWO different dialogs separated by dash line \'------\' for a given ' \
                'scene description (about {}):'

EXAMPLE = '[Example-{}]\n'

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError,
                                     openai.error.APIError, openai.error.APIConnectionError,
                                     openai.error.Timeout, openai.error.ServiceUnavailableError))
def completions_with_backoff(**kwargs):
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%Y-%m-%d")
    # system_content = f'Now you are a scenarist, and use your imagination to creat some conversations between a **Human** role and a **GPT** role under a given topic.\nKnowledge cutoff: 2021-09\nCurrent date:{currentTime}'
    system_content = f'You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent date:{currentTime}'
    print(system_content)
    return openai.ChatCompletion.create(
        model=kwargs['engine'],
        temperature=kwargs['temperature'],
        max_tokens=kwargs['max_tokens'],
        presence_penalty=kwargs['presence_penalty'],
        frequency_penalty=kwargs['frequency_penalty'],
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": kwargs['prompt']},
        ]
    )


@lru_cache(maxsize=1000)
def get_gpt_output(prompt, **kwargs):
    # gpt_logger.write(prompt)
    response = completions_with_backoff(prompt=prompt, engine=kwargs['engine'],
                                        temperature=kwargs['temperature'], max_tokens=kwargs['max_tokens'],
                                        presence_penalty=kwargs['presence_penalty'],
                                        frequency_penalty=kwargs['frequency_penalty'])

    response_str = response['choices'][0]['message']['content']
    gpt_logger.write(response_str)
    # gpt_logger.write('#' * 55)
    gpt_logger.write('*******************')
    return response_str


def parse_args():
    parser = argparse.ArgumentParser()

    # User options
    parser.add_argument('--exp', type=str, default='exp0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--shot_number', type=int, default=8, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=53, help='random seed')
    parser.add_argument('--resume', type=str, default='')

    # GPT settings
    parser.add_argument('--engine', type=str, default='gpt-3.5-turbo', choices=['text-davinci-002', 'gpt-3.5-turbo'])
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=768,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    parser.add_argument('--ckpt_root', type=str, default='./checkpoint/')

    args = parser.parse_args()

    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("_%Y_%m_%d_%H_%M_%S")
    args.exp = args.exp + currentTime

    # print and save the args
    args.ckpt_path = os.path.join(args.ckpt_root, args.exp)
    create_dir(args.ckpt_path)
    _logger = Logger(args.ckpt_path + '/args.txt')

    print('====Input Arguments====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))
    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    openai.api_key = 'sk-'
    # GPT parameters
    gpt_args = dict(
        engine=args.engine,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty
    )

    # ## TRAINING
    logger = Logger(os.path.join(args.ckpt_path, 'log.txt'))

    demos = get_demonstration('./IT-demos.txt')
    random.shuffle(demos)

    video_caption_path = '/data/T_X_pair_data/data/webvid/webvid.json'
    image_caption_path = '/data/T_X_pair_data//cc3m/cc3m.json'
    audio_caption_path = '/data/T_X_pair_data/audiocap/audiocap.json'

    sample_number = 5000
    modality = 'image'

    gpt_logger = Logger(os.path.join(args.ckpt_path, f'gpt_log_{modality}.txt'))
    if modality == 'video':
        captions = get_video_captions(video_caption_path)
    elif modality == 'image':
        captions = get_image_captions(image_caption_path)
    else:
        captions = get_audio_captions(audio_caption_path)
    captions = random.sample(captions, sample_number)
    save_captions(os.path.join(args.ckpt_path, f'captions_{modality}.txt'), captions)
    for cap in captions:
        demo_idx = random.sample(range(0, 16), 3)
        prompt = BEFORE_EXAMPLE + ''.join(
            [EXAMPLE.format(i + 1) + demos[idx] for i, idx in enumerate(demo_idx)]) + AFTER_EXAMPLE.format(modality)
        prompt += cap
        print(prompt)
        res = get_gpt_output(prompt, **gpt_args)
        if res:
            logger.write(modality + '\t' + cap)

