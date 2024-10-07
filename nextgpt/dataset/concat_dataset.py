from torch.utils.data import ConcatDataset, Dataset
from typing import List
import transformers
from .catalog import DatasetCatalog
from .dataset_utils import instantiate_from_config
from training_utils import DataArguments


class MyConcatDataset(Dataset):
    def __init__(self, 
                 dataset_name_list: List[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(MyConcatDataset, self).__init__()

        _datasets = []

        catalog = DatasetCatalog()
        for dataset_idx, dataset_name in enumerate(dataset_name_list):
            dataset_dict = getattr(catalog, dataset_name)

            target = dataset_dict['target']
            _params = dataset_dict['params']
            # print(target)
            
            params = dict()
            params['tokenizer'] = tokenizer
            params['data_args'] = data_args
            data_args.image_folder = _params.get('image_folder', None)
            data_args.image_caption_emb_folder = _params.get('image_caption_emb_folder', None)
            data_args.video_folder = _params.get('video_folder', None)
            data_args.video_caption_emb_folder = _params.get('video_caption_emb_folder', None)
            data_args.audio_folder = _params.get('audio_folder', None)
            data_args.audio_caption_emb_folder = _params.get('audio_caption_emb_folder', None)
            params['data_args'] = data_args
            params['data_path'] = _params.get('data_path', None)
            
            print(_params)
            dataset = instantiate_from_config(dict(target=target, params=params))

            _datasets.append(dataset)
        self.datasets = ConcatDataset(_datasets)
        print("cumulative_sizes: ", self.datasets.cumulative_sizes)  # [100, 200]

    def __len__(self):
        return self.datasets.__len__()

    def __getitem__(self, item):
        return self.datasets.__getitem__(item)
    
    @property
    def modality_lengths(self):
        length_list = []
        for dataset in self.datasets.datasets:
            print(dataset)
            length_list.extend(dataset.modality_lengths)
        return length_list
    
    def collate(self, instances):
        data = {key: [] for key in instances[0].keys()} if instances else {}

        for instance in instances:
            for key, value in instance.items():
                data[key].append(value)

        return data
    