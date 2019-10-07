import numpy as np

from stereoPair import stereoPair
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def get_train_loader(batch_size, path, device,
                     validation_size=0.1, shuffle=True, transform=None,
                     num_workers=0, pin_memory=False):
    dataset = stereoPair(path, device, transform=transform)
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_size*num_train))
    if shuffle:
        np.random.shuffle(indices)
    train_indices = indices[split:]
    val_indices = indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def get_test_loader(batch_size, path, device, transform=None,
                    num_workers=0, pin_memory=False):
    dataset = stereoPair(path, device, is_training=False, transform=transform)
    test_loader = DataLoader(dataset, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=pin_memory)
    return test_loader
