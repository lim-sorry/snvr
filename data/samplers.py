import torch


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


class EpochShuffleSampler(torch.utils.data.Sampler):
    r"""Shuffles the whole dataset once per epoch with a generator seeded by seed + epoch.

    Gives the same order as torch's DistributedSampler with a single replica, so runs
    stay reproducible against those made with the earlier distributed code.
    """

    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=g).tolist())

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = epoch