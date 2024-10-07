import json
import os
from datetime import datetime
import backoff
import openai
import argparse
from functools import lru_cache
from transformers import logging

logging.set_verbosity_error()
from utils import *

# from base_prompt import build_prompt


BEFORE_EXAMPLE = 'Now you are a scenarist, and use your imagination to creat some dialogs between a **Human** ' \
                 'role and a **GPT** role under a the topic of **{}**. '

AFTER_EXAMPLE = {
    'T_A': 'Please note that the generation of dialog should meet the following rules:\n'
           '1. Ensure that the generated dialogue includes text and audio modalities. '
           '\'<Audio> Audio content description </Audio>\' is the placeholder that represents the audio content.\n'
           '2. Keep the number of dialogue interactions no more than {} rounds between **Human** and a **GPT**.\n'
           '3. Ensure that the content, form and style of dialogues are diverse among others, and use first '
           'person, second or third person perspective for the subject, and do not ask similar questions in '
           'different utterances, and also do NOT entirely copy the provided demonstration.\nPlease '
           'follow the above instructions, and produce {} dialogues, with each dialogue separated by dash '
           'line \'------\'.\nHere are examples:\n',
    'T_A_I': 'Please note that the generation of dialog should meet the following rules:\n'
             '1. Keep the number of dialogue interactions no more than {} rounds between **Human** and a **GPT**.\n'
             '2. Ensure that the generated dialogue includes text, audios and image modalities of content. '
             'Use two types of special markers as placeholders to represent the content of **audios** and '
             '**images** modalities:\n a. : \'<Audio> Audio content description </Audio>\' represents the audio '
             'content.\n b. : \'<Image> Image content description </Image>\' represents the image content.\n'
             '3. Ensure that the content, form and style of dialogues are diverse among others, and use first '
             'person, second or third person perspective for the subject, and do not ask similar questions in '
             'different utterances, and also do NOT entirely copy the provided demonstration.\nPlease '
             'follow the above instructions, and produce {} dialogues, with each dialogue separated by dash '
             'line \'------\'.\nHere are examples:\n',
    'T_A_V': 'Please note that the generation of dialog should meet the following rules:\n'
             '1. Keep the number of dialogue interactions no more than {} rounds between **Human** and a **GPT**.\n'
             '2. Ensure that the generated dialogue includes text, videos and audios modalities of content. '
             'Use two types of special markers as placeholders to represent the content of **audios** and '
             '**videos** modalities:\n a. : \'<Audio> Audio content description </Audio>\' represents the audio '
             'content.\n b. : \'<Video> Video content description </Video>\' represents the video content.\n'
             '3. Ensure that the content, form and style of dialogues are diverse among others, and use first '
             'person, second or third person perspective for the subject, and do not ask similar questions in '
             'different utterances, and also do NOT entirely copy the provided demonstration.\nPlease '
             'follow the above instructions, and produce {} dialogues, with each dialogue separated by dash '
             'line \'------\'.\nHere are examples:\n',
    'T_I': 'Please note that the generation of dialog should meet the following rules:\n'
           '1. Keep the number of dialogue interactions no more than {} rounds between **Human** and a **GPT**.\n'
           '2. Ensure that the generated dialogue includes text and images modalities of content. '
           'Use \'<Image> Image content description </Image>\' as placeholders to represent the image content.\n'
           '3. Ensure that the content, form and style of dialogues are diverse among others, and use first '
           'person, second or third person perspective for the subject, and do not ask similar questions in '
           'different utterances, and also do NOT entirely copy the provided demonstration.\nPlease '
           'follow the above instructions, and produce {} dialogues, with each dialogue separated by dash '
           'line \'------\'.\nHere are examples:\n',
    'T_I_V_A': 'Please note that the generation of dialog should meet the following rules:\n'
               '1. Keep the number of dialogue interactions no more than {} rounds between **Human** and a **GPT**.\n'
               '2. Ensure that the generated dialogue includes various modalities of content, such as text, images, '
               'videos, or audios. Use three types of special markers as placeholders to represent the  content of '
               '**images**, **videos** and **audio** modalities:\n a. \'<Image> Image content description </Image>\' '
               'represents the image content.\n b. \'<Video> Video content description </Video>\' represents the '
               'video content.\n c. \'<Audio> Audio content description </Audio>\' represents the audio content.\n'
               '3. Ensure that the content, form and style of dialogues are diverse among others, and use first '
               'person, second or third person perspective for the subject, and do not ask similar questions in '
               'different utterances, and also do NOT entirely copy the provided demonstration.\nPlease '
               'follow the above instructions, and produce {} dialogues, with each dialogue separated by dash '
               'line \'------\'.\nHere are examples:\n',
    'T_V': 'Please note that the generation of dialog should meet the following rules:\n'
           '1. Keep the number of dialogue interactions no more than {} rounds between **Human** and a **GPT**.\n'
           '2. Ensure that the generated dialogue includes text and videos modalities of content. '
           'Use \'<Video> Video content description </Video>\' as placeholders to represent the video content.\n'
           '3. Ensure that the content, form and style of dialogues are diverse among others, and use first '
           'person, second or third person perspective for the subject, and do not ask similar questions in '
           'different utterances, and also do NOT entirely copy the provided demonstration.\nPlease '
           'follow the above instructions, and produce {} dialogues, with each dialogue separated by dash '
           'line \'------\'.\nHere are examples:\n',
    'T_V_I': 'Please note that the generation of dialog should meet the following rules:\n'
             '1. Keep the number of dialogue interactions no more than {} rounds between **Human** and a **GPT**.\n'
             '2. Ensure that the generated dialogue includes text, videos and images modalities of content. '
             'Use two types of special markers as placeholders to represent the content of **videos** and '
             '**images** modalities:\n a. : \'<Video> Video content description </Video>\' represents the video '
             'content.\n b. : \'<Image> Image content description </Image>\' represents the image content.\n'
             '3. Ensure that the content, form and style of dialogues are diverse among others, and use first '
             'person, second or third person perspective for the subject, and do not ask similar questions in '
             'different utterances, and also do NOT entirely copy the provided demonstration.\nPlease '
             'follow the above instructions, and produce {} dialogues, with each dialogue separated by dash '
             'line \'------\'.\nHere are examples:\n'
}

SEPERATE_LINE = '------\n'

# TOPICS = ['breakfast', 'lunch', 'school canteen', 'fitness', 'family time', 'movie', 'book', 'travel', 'food',
#           'social media', 'weather forecast', 'commute', 'shopping', 'laundry', 'pets', 'meeting', 'coffee',
#           'milk tea', 'health', 'hospital', 'school', 'gardening', 'volunteering', 'research work', 'english learning',
#           'Artificial Intelligence', 'animals', 'zoo', 'plants', 'festival', 'novels', 'market', 'grocery',
#           'environment', 'costume', 'fashion', 'photography', 'painting', 'Theater Productions', 'TV series', 'athlete',
#           'history', 'symphony', 'pop music', 'investment', 'finance', 'education', 'yoga', 'jogging', 'badminton']

# TOPICS = ['friend', 'vegetable', 'fruit', 'seafood', 'credit card', 'transportation', 'discount', 'make-up', 'taxi',
#           'take a flight', 'city walk', 'ticket', 'Pharmacy', 'concert', 'museum', 'internet', 'Stress management',
#           'smartphone', '5G', 'wearable tech', 'banking', 'savings', 'loan', 'adventure', 'backpacking',
#           'Pollution', 'marriage', 'online dating', 'furniture', 'decoration', 'pet care', 'home security', 'DIY',
#           'social networking', 'interpersonal skill', 'sleep quality', 'kitchen', 'electronic shoping', 'car pooling',
#           'board games', 'Amusement parks', 'online course', 'Educational resources', 'job interview',
#           'workplace culture', 'Household chores', 'Home appliances', 'Social etiquette', 'Cultural diversity',
#           'Traditions and customs', 'Car rentals', 'Picnic', 'Birdwatching']


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
    gpt_logger.write('------')
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
    parser.add_argument('--engine', type=str, default='gpt-4', choices=['text-davinci-002', 'gpt-3.5-turbo', 'gpt-4 turbo'])
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=768,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    parser.add_argument('--ckpt_root', type=str, default='./checkpoint/demonstrations_1')

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
    gpt_logger = Logger(os.path.join(args.ckpt_path, 'gpt_log.txt'))
    TOPICS = load_topics(os.path.join('./demonstrations', 'more_topics.txt'))
    round_num = 4
    dialogue_num = 1
    mode = ['T_A', 'T_A_I', 'T_I', 'T_I_V_A', 'T_V', 'T_V_I']
    existing_states, newest_directory = get_existing_states(args.ckpt_path)
    if newest_directory is not None:
        with open(os.path.join(args.ckpt_path, newest_directory, 'gpt_log.txt'), 'r', encoding='utf-8') as f:
            for lines in f.readlines():
                gpt_logger.write(lines)
    for m in mode:
        demonstrations = get_demonstration(os.path.join('./demonstrations', f'{m}.txt'))
        for idx, d in enumerate(demonstrations):
            # for ydx, dem in enumerate(demonstrations[idx+1:]):
            for topic in TOPICS:
                current_state = f'{m}_{idx}_{topic}'
                # logger.write(current_state)
                if current_state not in existing_states:
                    promt = BEFORE_EXAMPLE.format(topic) + AFTER_EXAMPLE[m].format(round_num, dialogue_num) + d
                    print(promt)
                    res = get_gpt_output(promt, **gpt_args)
                    if res:
                        logger.write(current_state)
                else:
                    logger.write(current_state)
            # exit(0)