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


class TX2TInstructionDataset(Dataset):
    """
    T + X - T instruction Dataset
    """
    def __init__(self, data_path: str, mm_root_path: str = None, dataset_type: str='ImageToText'):
        super(TX2TInstructionDataset, self).__init__()

        self.mm_root_path = mm_root_path
        self.instruction_list = []
        self.mm_path_list = []
        self.dataset_category = 't2t' if mm_root_path is None else 'tx2t'
        with open(data_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        for instance in tqdm(res, total=len(res)):
            self.instruction_list.append(instance['conversation'])
            if self.dataset_category == 'tx2t':
                # Text + X -> Text dataset
                self.mm_path_list.append(os.path.join(mm_root_path, instance['image_name']))
        self.dataset_type_list = [dataset_type for _ in range(len(self.instruction_list))]

    def __len__(self):  # number of instances
        return len(self.instruction_list)

    def __getitem__(self, i):
        if self.dataset_category == 'tx2t':
            # Text + X -> Text dataset
            return dict(mm_paths=self.mm_path_list[i], output_texts=self.instruction_list[i],
                        dataset_types=self.dataset_type_list[i])
        else:
            # Text -> Text dataset
            return dict(output_texts=self.instruction_list[i], dataset_types=self.dataset_type_list[i])

    def collate(self, instances):
        if self.dataset_category == 'tx2t':
            mm_paths, output_texts, dataset_types = tuple(
                [instance[key] for instance in instances] for key in ("mm_paths", "output_texts", "dataset_types"))
            return dict(
                mm_paths=mm_paths,
                output_texts=output_texts,
                dataset_types=dataset_types
            )
        else:
            output_texts, dataset_types = tuple(
                [instance[key] for instance in instances] for key in ("output_texts", "dataset_types"))
            return dict(
                output_texts=output_texts,
                dataset_types=dataset_types
            )
