import os
import pickle
import numpy as np
import argparse

import config
from utilities import plot_class_wise_accuracy


def get_average_cv_stats(args):
    
    workspace = args.workspace
    filename = args.filename
    iteration = args.iteration
    
    labels = config.labels
    
    pickles_dir = os.path.join(workspace, 'stats', filename)
    
    correctness_list = []
    total_list = []
    accuracy_list = []
    mapk_list = []
    
    for holdout_fold in [1, 2, 3, 4]:
        
        pickle_path = os.path.join(pickles_dir, 'holdout_fold{}'.format(holdout_fold), '{}_iters.p'.format(iteration))
        dict = pickle.load(open(pickle_path, 'rb'))
        correctness_list.append(dict['correctness'])
        total_list.append(dict['total'])
        accuracy_list.append(dict['accuracy'])
        mapk_list.append(dict['mapk'])
        
    correctness = np.sum(correctness_list, axis=0)
    total = np.sum(total_list, axis=0)
    
    class_wise_accuracy = correctness / total.astype(np.float32)
    
    # Print
    print('accuracy: {}'.format(np.mean(accuracy_list)))
    print('mapk: {}'.format(np.mean(mapk_list)))
    
    print('{:<30}{}/{}\t{}'.format('event labels', 'correct', 'total', 'accuracy'))
    for (n, label) in enumerate(labels):
        print('{:<30}{}/{}\t\t{:.2f}'.format(label, correctness[n], total[n], class_wise_accuracy[n]))
        
    # Plot class wise accuracy
    plot_class_wise_accuracy(class_wise_accuracy)
        
        
if __name__ == '__main__':
    """Load the result of cross validation and summarize to a single result. 
    
    Example: python utils/get_average_cv_stats.py --workspace=$WORKSPACE --filename=main_pytorch --iteration=3000
    """
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--iteration', type=str, required=True)
    
    args = parser.parse_args()
    
    get_average_cv_stats(args)