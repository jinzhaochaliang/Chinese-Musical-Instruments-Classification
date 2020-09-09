import os
import numpy
import argparse
import sys
import soundfile
import numpy as np
import librosa
import h5py
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import logging

import config
    
    
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
   
   
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na
   
    
def create_logging(log_dir, filemode):
    
    create_folder(log_dir)
    i1 = 0
    
    while os.path.isfile(os.path.join(log_dir, "%04d.log" % i1)):
        i1 += 1
        
    log_path = os.path.join(log_dir, "%04d.log" % i1)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
                
    # Print to console   
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging
   

def read_audio(path, target_fs=None):

    (audio, fs) = soundfile.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs


def calculate_scalar(x):

    if x.ndim == 2:
        axis = 0
        
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def scale(x, mean, std):

    return (x - mean) / std


def inverse_scale(x, mean, std):

    return x * std + mean


def repeat_seq(x, time_steps):
    repeat_num = time_steps // len(x) + 1
    repeat_x = np.tile(x, (repeat_num, 1))[0 : time_steps]
    return repeat_x
    
    
def calculate_accuracy(output, target):
    acc = np.sum(output == target) / float(len(target))
    return acc
    

def print_class_wise_accuracy(output, target):
    """Print class wise accuracy."""
    
    labels = config.labels
    ix_to_lb = config.ix_to_lb
    
    correctness = np.zeros(len(labels), dtype=np.int32)
    total = np.zeros(len(labels), dtype=np.int32)
  
    for n in range(len(target)):
        
        total[target[n]] += 1
        
        if output[n] == target[n]:
            correctness[target[n]] += 1
        
    class_wise_accuracy = correctness.astype(np.float32) / total
    
    logging.info('{:<30}{}/{}\t{}'.format(
        'event labels', 'correct', 'total', 'accuracy'))
        
    for (n, label) in enumerate(labels):
        logging.info('{:<30}{}/{}\t\t{:.2f}'.format(
            label, correctness[n], total[n], class_wise_accuracy[n]))
        
    class_wise_accuracy = np.array(class_wise_accuracy)
    
    return class_wise_accuracy, correctness, total

    
def plot_class_wise_accuracy(class_wise_accuracy):
    """Plot accuracy."""
    
    labels = config.labels
    classes_num = len(labels)
    
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4))
    ax.bar(np.arange(classes_num), class_wise_accuracy, alpha=0.5)
    ax.set_ylabel('Accuracy')
    ax.set_xlim(0, classes_num)
    ax.set_ylim(0., 1.)
    ax.xaxis.set_ticks(np.arange(classes_num))
    ax.xaxis.set_ticklabels(labels, rotation=90)
    plt.tight_layout()
    plt.show()
    
    
def write_testing_data_submission_csv(submission_path, audio_names, 
                                      sorted_indices):
    
    kmax = config.kmax
    ix_to_lb = config.ix_to_lb
    corrupted_files = config.corrupted_files
    
    # Write result to submission csv
    f = open(submission_path, 'w')
    
    f.write('fname,label\n')
    
    for (n, audio_name) in enumerate(audio_names):
        
        f.write('{},'.format(audio_name))
        
        predicted_labels = [ix_to_lb[sorted_indices[n, k]] for k in range(kmax)]
        
        f.write(' '.join(predicted_labels))
            
        f.write('\n')
    
    for audio_name in corrupted_files:
        f.write('{},{}\n'.format(audio_name, 'Acoustic_guitar'))
    
    f.close()
    
    print("Write result to {}".format(submission_path))
