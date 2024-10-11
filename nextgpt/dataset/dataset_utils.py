
from nextgpt.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, MAX_IMAGE_LENGTH
from nextgpt.constants import VIDEO_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, MAX_VIDEO_LENGTH
from nextgpt.constants import AUDIO_TOKEN_INDEX, DEFAULT_AUDIO_TOKEN, DEFAULT_AUD_START_TOKEN, DEFAULT_AUD_END_TOKEN, MAX_AUDIO_LENGTH
from nextgpt import conversation as conversation_lib
from nextgpt.mm_utils import tokenizer_image_token, tokenizer_multiple_token
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast, Sequence
from training_utils import DataArguments
import copy
import tokenizers
import transformers
import torch
import torch.nn as nn
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
import re
import importlib


def process_caption(caption):
    caption = re.sub(
        r"([\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    return caption


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_other_modality: bool = True,  # None, 'image', 'video', 'audio', 'mixed'
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer, has_other_modality=has_other_modality)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_other_modality=has_other_modality)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_other_modality=has_other_modality)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_other_modality=has_other_modality)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_multiple_token(prompt, tokenizer)) for prompt in prompts]

    if has_other_modality:
        input_ids = [tokenizer_multiple_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_other_modality:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for idx, sentence in enumerate(source):
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')

                IMAGE_TOKEN_NUM = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                if IMAGE_TOKEN_NUM > MAX_IMAGE_LENGTH:
                    sentence['value'] = sentence['value'].replace((DEFAULT_IMAGE_TOKEN + ' ') * IMAGE_TOKEN_NUM, DEFAULT_IMAGE_TOKEN * MAX_IMAGE_LENGTH).strip()
                
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_img_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # if the GPT output image or video or audio, it should be replaced by the signal tokens.
            if idx % 2 == 1:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, ' '.join([f"<image_{i:02d}>" for i in range(data_args.n_img_tokens)]))

            if DEFAULT_VIDEO_TOKEN in sentence['value']:
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, '<Video>' + DEFAULT_VIDEO_TOKEN + '</Video>')
                VIDEO_TOKEN_NUM = sentence['value'].count(DEFAULT_VIDEO_TOKEN)
                if VIDEO_TOKEN_NUM > MAX_VIDEO_LENGTH:
                    sentence['value'] = sentence['value'].replace((DEFAULT_VIDEO_TOKEN + ' ') * VIDEO_TOKEN_NUM, DEFAULT_VIDEO_TOKEN * MAX_VIDEO_LENGTH).strip()
                
            replace_token = DEFAULT_VIDEO_TOKEN
            if data_args.mm_use_vid_start_end:
                replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, replace_token)

            if idx % 2 == 1:
                sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, ' '.join([f"<video_{i:02d}>" for i in range(data_args.n_vid_tokens)]))

            if DEFAULT_AUDIO_TOKEN in sentence['value']:
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_AUDIO_TOKEN, '<Audio>' + DEFAULT_AUDIO_TOKEN + '</Audio>')
                AUDIO_TOKEN_NUM = sentence['value'].count(DEFAULT_AUDIO_TOKEN)
                if AUDIO_TOKEN_NUM > MAX_AUDIO_LENGTH:
                    sentence['value'] = sentence['value'].replace((DEFAULT_AUDIO_TOKEN + ' ') * AUDIO_TOKEN_NUM, DEFAULT_AUDIO_TOKEN * MAX_AUDIO_LENGTH).strip()
                
            replace_token = DEFAULT_AUDIO_TOKEN
            if data_args.mm_use_aud_start_end:
                replace_token = DEFAULT_AUD_START_TOKEN + replace_token + DEFAULT_AUD_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_AUDIO_TOKEN, replace_token)

            if idx % 2 == 1:
                sentence["value"] = sentence["value"].replace(DEFAULT_AUDIO_TOKEN, ' '.join([f"<audio_{i:02d}>" for i in range(data_args.n_aud_tokens)]))

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_other_modality: bool = True,  # None, 'image', 'video', 'audio', 'mixed'
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_other_modality:
        input_ids = torch.stack([tokenizer_multiple_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], axis=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_other_modality:
                round_len = len(tokenizer_multiple_token(rou, tokenizer))
                instruction_len = len(tokenizer_multiple_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_other_modality: bool = True,  # None, 'image', 'video', 'audio', 'mixed'
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_other_modality:
        # print(conversations)
        input_ids = torch.stack([tokenizer_multiple_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], axis=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_other_modality:
                round_len = len(tokenizer_multiple_token(rou, tokenizer))
                instruction_len = len(tokenizer_multiple_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_other_modality: bool = True
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_other_modality:
        input_ids = torch.stack([tokenizer_multiple_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], axis=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_other_modality:
                round_len = len(tokenizer_multiple_token(rou, tokenizer))
                instruction_len = len(tokenizer_multiple_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_other_modality: bool = True,  # None, 'image', 'video', 'audio', 'mixed'
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    # print("sources: ", sources)
    for source in sources:
        assert len(source) == 2
        conversation = source[0]['value'] + source[1]['value'] + '' + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    if has_other_modality:
        input_ids = [tokenizer_multiple_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_multiple_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)



def pad_sequence(sequences: List[torch.Tensor],
                 batch_first: bool=False,
                 padding_value: float=0.0) -> torch.Tensor:
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from paddle.nn.utils.rnn import pad_sequence
        >>> a = paddle.ones(25, 300)
        >>> b = paddle.ones(22, 300)
        >>> c = paddle.ones(15, 300)
        >>> pad_sequence([a, b, c]).shape
        paddle.Tensor([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = torch.shape(sequences[0])
    # (TODO Hui Zhang): slice not supprot `end==start`
    # trailing_dims = max_size[1:]
    trailing_dims = tuple(
        max_size[1:].numpy().tolist()) if sequences[0].ndim >= 2 else ()
    max_len = max([s.shape[0] for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        # use index notation to prevent duplicate references to the tensor
        # logger.debug(
        #     f"length {length}, out_tensor {out_tensor.shape}, tensor {tensor.shape}"
        # )
        if batch_first:
            # TODO (Hui Zhang): set_value op not supprot `end==start`
            # TODO (Hui Zhang): set_value op not support int16
            # TODO (Hui Zhang): set_varbase 2 rank not support [0,0,...]
            # out_tensor[i, :length, ...] = tensor
            if length != 0:
                out_tensor[i, :length] = tensor
            else:
                out_tensor[i, length] = tensor
        else:
            # TODO (Hui Zhang): set_value op not supprot `end==start`
            # out_tensor[:length, i, ...] = tensor
            if length != 0:
                out_tensor[:length, i] = tensor
            else:
                out_tensor[length, i] = tensor

    return out_tensor