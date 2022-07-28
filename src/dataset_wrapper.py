from itertools import repeat

import torch


def _repeater(dataloader):
    for loader in repeat(dataloader):
        for data in loader:
            yield data


def _dataloader(data, sampler, batch_size, num_workers, use_cpu):
    if use_cpu:
        pin_memory = False 
        num_workers = 0 
    else:
        pin_memory = True 
        
    return torch.utils.data.DataLoader(
            data, batch_size=batch_size, num_workers=num_workers,
            sampler=sampler, drop_last=True, pin_memory=pin_memory)


def infinite_dataloader(data, sampler, batch_size, num_workers, use_cpu):
    return _repeater(_dataloader(data, sampler, batch_size, num_workers, use_cpu))


class Wrap(object):
    def __init__(self, data, sampler, batch_size, samples, num_workers, use_cpu):
        self.data = infinite_dataloader(data, sampler, batch_size, num_workers, use_cpu)
        self.samples = samples

    def __iter__(self):
        for i in range(self.samples):
            yield next(self.data)

    def __len__(self):
        return self.samples
