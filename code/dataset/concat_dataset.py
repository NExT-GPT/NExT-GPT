from torch.utils.data import ConcatDataset, Dataset
from .catalog import DatasetCatalog
from .utils import instantiate_from_config


class MyConcatDataset(Dataset):
    def __init__(self, dataset_name_list):
        super(MyConcatDataset, self).__init__()

        _datasets = []

        catalog = DatasetCatalog()
        for dataset_idx, dataset_name in enumerate(dataset_name_list):
            dataset_dict = getattr(catalog, dataset_name)

            target = dataset_dict['target']
            params = dataset_dict['params']
            print(target)
            print(params)
            dataset = instantiate_from_config(dict(target=target, params=params))

            _datasets.append(dataset)
        self.datasets = ConcatDataset(_datasets)

    def __len__(self):
        return self.datasets.__len__()

    def __getitem__(self, item):
        return self.datasets.__getitem__(item)

    def collate(self, instances):
        data = {key: [] for key in instances[0].keys()} if instances else {}

        for instance in instances:
            for key, value in instance.items():
                data[key].append(value)

        return data
