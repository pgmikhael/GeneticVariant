from os.path import dirname, realpath, join
import sys
sys.path.append((dirname(dirname(realpath(__file__)))))
from datasets.dataset_factory import get_dataset
from models.model_factory import get_model, load_model, get_rolled_out_size
from learning.learn import train_model, eval_model
from helpers.command_methods import get_dataset_stats
import git
import torch
import pdb
import helpers.parsing as parsing
import warnings
import time
import pickle
import numpy as np

#Constants
DATE_FORMAT_STR = "%m-%d-%Y %H:%M:%S"
RESULTS_DATE_FORMAT = "%m%d%Y-%H%M%S"
SEED = 9197

torch.manual_seed(SEED)
np.random.seed(SEED)

if __name__ == '__main__':
    args = parsing.parse_args()
    # print args
    for key,value in vars(args).items():
        print('{} -- {}'.format(key.upper(), value))
    
    repo = git.Repo(search_parent_directories=True)
    commit  = repo.head.object
    print("\nGenetic Variant main running by author: {} \ndate:{}, \nfrom commit: {} -- {}".format(
        commit.author, time.strftime(DATE_FORMAT_STR, time.localtime(commit.committed_date)),
        commit.hexsha, commit.message))

    if args.get_dataset_stats:
        print("\nComputing sequence length...")
        args.seq_len = get_dataset_stats(args)
        print('Sequence length = {}'.format(args.seq_len))

    # Obtain datasets
    if args.train_phase:
        print("\nLoading train and dev data...")
        train_data = get_dataset(args, 'train')
        dev_data = get_dataset(args, 'dev')

    if args.test_phase:
        if not args.train_phase:
            print("\nLoading dev data...")
            dev_data = get_dataset(args, 'dev')
        print("\nLoading test data...")
        test_data = get_dataset(args, 'test')
    

    # Load model
    model, optimizer = get_model(args)
    if args.snapshot_path is not None:
        try:
            model, optimizer, args.lr, args.epoch_stats = load_model(args.snapshot_path, model, optimizer, args)
        except:
            print("\n Error loading snapshot...Starting run from scratch.")
    else:
        args.epoch_stats = None
    print(model)
    
    args.run_time = time.strftime(RESULTS_DATE_FORMAT, time.localtime())

    # Train model and get statistics
    model_stats = {}
    if args.train_phase:
        print("\nBeginning Training Phase:")
        model_stats['train_stats'], model, optimizer = train_model(train_data, dev_data, model, optimizer, args)

    # Test model and get statistics
    if args.test_phase:
        print("\nBeginning Testing Phase:")
        model_stats['dev_stats'] = eval_model(dev_data, model, optimizer, 'dev', args)
        model_stats['test_stats'] = eval_model(test_data, model, optimizer, 'test', args)


    # Save results
    if args.results_path is not None:
        save_path = args.results_path
    else:
        save_path = join(args.save_dir, "{}_{}_{}".format(args.model_name, args.dataset, args.run_time) )

    pickle.dump(model_stats, open("{}.rslt".format(save_path), 'wb'))
    pickle.dump(vars(args),  open("{}.args".format(save_path), 'wb'))

    print("Saved: \nresults to {} \nargs to {}".format( \
        "{}.rslt".format(save_path), \
        "{}.args".format(save_path)) \
    )