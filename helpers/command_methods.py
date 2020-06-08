import torch
import os.path
from datasets.dataset_factory import get_dataset
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

def get_dataset_stats(args):
    args = copy.deepcopy(args)
    args.computing_stats = True
    train = get_dataset(args, 'train')
    string_lens = [len(row['x']) for row in train.dataset]
    return max(string_lens)

