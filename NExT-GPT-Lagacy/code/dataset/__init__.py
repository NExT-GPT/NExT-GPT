from header import *
from .samplers import DistributedBatchSampler, DistributedMultiDatasetBatchSampler
from .catalog import DatasetCatalog
from .utils import instantiate_from_config
import torch
from torch.utils.data import ConcatDataset
from .concat_dataset import MyConcatDataset


def load_dataset(args, dataset_name_list):
    """
    Args:
        args:
        dataset_name_list: List[str]
        repeats: List[int], the training epochs for each dataset

    """
    # concat_data = get_concat_dataset(dataset_name_list)
    concat_data = MyConcatDataset(dataset_name_list)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    sampler = torch.utils.data.RandomSampler(concat_data)
    batch_sampler = DistributedMultiDatasetBatchSampler(dataset=concat_data,
                                                        sampler=sampler,
                                                        batch_size=batch_size,
                                                        drop_last=True,
                                                        rank=rank,
                                                        world_size=world_size)
    iter_ = DataLoader(
        concat_data,
        batch_sampler=batch_sampler,
        num_workers=1,
        collate_fn=concat_data.collate,
        pin_memory=True
    )
    return concat_data, iter_, sampler
