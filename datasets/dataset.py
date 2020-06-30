import torch
from torch.utils import data
from datasets.dataset_factory import RegisterDataset
import json
from collections import defaultdict
from tqdm import tqdm
import time
import os
from copy import copy
from datasets.abstract_dataset import Abstract_Dataset
import string 

ALL_LETTERS = string.punctuation + string.ascii_letters + string.digits
NUM_ALL_LETTERS = len(ALL_LETTERS)

METADATA_FILENAMES = {
    "baseline_prediction": "variant_classification_dataset.json",
    "multi_split": "variant_classification_dataset_multisplits.json"}

@RegisterDataset("variant_names")
class GeneticVariants(Abstract_Dataset):

    def create_dataset(self, split_group):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        """
        dataset = []
        for row in tqdm(self.metadata_json, position=0):
            str_id, split, string, y, label =  row['id'], row['split_{}'.format(self.args.split_num)], row['x'], row['y'], row['label']
            original_str_ln = len(string)
            if not split == split_group:
                continue

            if len(string) > self.args.max_str_len:
                if self.args.truncate_string:
                    string = string[:self.args.max_str_len]
                else:
                    continue
            
            if not self.args.computing_stats:
                x = self.pad_tensor(strToTensor(string))
            else:
                x = string

            dataset.append({
                'id': str_id,
                'x': x,
                'y': y,
                'label': label,
                'string': string,
                'string_lens': len(string),
                'original_str_ln': original_str_ln
            })
        
        #dataset = sorted(dataset, key = lambda row: row['string_lens'], reverse=True)
        return dataset


    def get_label(self, post, task):
        pass

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAMES[self.task]

    @staticmethod
    def set_args(args):
        args.num_classes = 3
        args.input_dim = NUM_ALL_LETTERS

    @property
    def task(self):
        return self.args.task # "prediction"

    def pad_tensor(self, tensor):
        pad_tensor = torch.zeros(self.args.seq_len - tensor.shape[0], NUM_ALL_LETTERS)
        return torch.cat([tensor,pad_tensor], dim = 0 )


def strToTensor(line):
    tensor = torch.zeros(len(line), NUM_ALL_LETTERS)
    for letter_index, letter in enumerate(line):
        one_hot_index = ALL_LETTERS.find(letter)
        tensor[letter_index, one_hot_index] = 1
    return tensor

