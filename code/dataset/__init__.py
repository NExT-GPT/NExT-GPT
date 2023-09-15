from header import *
from .samplers import DistributedBatchSampler


def load_dataset(args, dataset_name):
    """
    Args:
        dataset_name: str
        data_path : str,  args['data_path']
        mm_root_path: str, args['mm_root_path']
    """
    if dataset_name == 'cc3m':
        from .cc3m_dataset import CC3MDataset
        data = CC3MDataset(args['data_path'], args['mm_root_path'], args['embed_path'])
    elif dataset_name == 'webvid':
        from .webvid_dataset import WebvidDataset
        data = WebvidDataset(args['data_path'], args['mm_root_path'], args['embed_path'])
    elif dataset_name == 'audiocap':
        from .audiocap_dataset import AudioCapDataset
        data = AudioCapDataset(args['data_path'], args['mm_root_path'], args['embed_path'])
    elif dataset_name == 'instruction':
        from .instruction_dataset import InstructionDataset
        data = InstructionDataset(args['data_path'], args['mm_root_path'], args['embed_path'])
    else:
        raise NotImplementedError

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler,
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data,
        batch_sampler=batch_sampler,
        num_workers=1,
        collate_fn=data.collate,
        pin_memory=True
    )
    return data, iter_, sampler
