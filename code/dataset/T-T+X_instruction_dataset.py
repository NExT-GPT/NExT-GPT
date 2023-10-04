import json
import os.path

from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import re
import random
import numpy as np
import torch


# from .base_dataset import BaseDataset


class T2XTInstructionDataset(Dataset):
    """
        T - T + X instruction Dataset
        """
    def __init__(self, data_path: str, embed_path: str, dataset_type: str = "TextToImage"):
        super(T2XTInstructionDataset, self).__init__()

        self.embed_path = embed_path
        self.instruction_list = []
        self.mm_path_list = []
        with open(data_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        for instance in tqdm(res, total=len(res)):
            self.instruction_list.append(instance['conversation'])
            self.mm_path_list.append(instance['mm_name'])
        self.dataset_type_list = [dataset_type for _ in range(len(self.instruction_list))]

    def __len__(self):  # number of instances
        return len(self.instruction_list)

    def __getitem__(self, i):
        with open(os.path.join(self.embed_path, str(os.path.basename(self.mm_path_list[i])) + '.npy'), 'rb') as f:
            caption_embs = torch.from_numpy(np.load(f, allow_pickle=True))  # (num_clip_tokens, 768)

        return dict(output_texts=self.instruction_list[i], caption_embs=caption_embs, dataset_types=self.dataset_type_list[i])

    def collate(self, instances):
        output_texts, caption_embs, dataset_types = tuple(
            [instance[key] for instance in instances] for key in ("output_texts", "caption_embs", "dataset_types"))
        return dict(
            output_texts=output_texts,
            caption_embs=caption_embs,
            dataset_types=dataset_types
        )
