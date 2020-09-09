import os
import sys
import numpy as np
import pandas as pd
import argparse
import random

import config


def create_validation_folds(args):
    """Create validation file with folds and write out to validate_meta.csv
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    
    labels = config.labels
    random_state = np.random.RandomState(1234)
    folds_num = 4
    
    # Paths
    csv_path = os.path.join(dataset_dir, 'train.csv')
    
    # Read csv
    df = pd.DataFrame(pd.read_csv(csv_path))
    
    indexes = np.arange(len(df))
    random_state.shuffle(indexes)
    
    audios_num = len(df)
    audios_num_per_fold = int(audios_num // folds_num)

    # Create folds
    folds = np.zeros(audios_num, dtype=np.int32)
    
    for n in range(audios_num):
        folds[indexes[n]] = (n % folds_num) + 1

    df_ex = df
    df_ex['fold'] = folds

    # Write out validation csv
    out_path = os.path.join(workspace, 'validate_meta.csv')
    df_ex.to_csv(out_path)

    print("Write out to {}".format(out_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir')
    parser.add_argument('--workspace')

    args = parser.parse_args()

    create_validation_folds(args)
