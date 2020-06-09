from torch.utils import data
import torch

def ignore_None_collate(batch):
    '''
    dataloader.default_collate wrapper that creates batches only of not None values.
    Useful for cases when the dataset.__getitem__ can return None because of some
    exception and then we will want to exclude that sample from the batch.
    '''
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return data.dataloader.default_collate(batch)

def get_train_and_dev_dataset_loaders(args, train_data, dev_data):
    '''
        Given arg configuration, return appropriate torch.DataLoader
        for train_data and dev_data

        returns:
        train_data_loader: iterator that returns batches
        dev_data_loader: iterator that returns batches
    '''
    if args.class_bal:
        sampler = data.sampler.WeightedRandomSampler(
                weights=train_data.weights,
                num_samples=len(train_data),
                replacement=True)

        train_data_loader = data.DataLoader(
                train_data,
                num_workers=args.num_workers,
                sampler=sampler,
                pin_memory=True,
                batch_size=args.batch_size,
                collate_fn=ignore_None_collate)
    else:
        train_data_loader = data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=ignore_None_collate,
            pin_memory=True,
            drop_last=True)

    dev_data_loader = data.DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ignore_None_collate,
        pin_memory=True,
        drop_last=False)

    return train_data_loader, dev_data_loader

def l1_regularization(model):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.norm(param, p=1) # torch.sum(torch.abs(param))
    return regularization_loss
