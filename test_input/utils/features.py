import os
import sys
import numpy
import argparse
import sys
import soundfile
import numpy as np
import librosa
import h5py
import time
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import logging

import config
from utilities import calculate_scalar, create_folder, read_audio


class LogMelExtractor():
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np.hamming(window_size)
        
        self.melW = librosa.filters.mel(sr=sample_rate, 
                                        n_fft=window_size, 
                                        n_mels=mel_bins, 
                                        fmin=50., 
                                        fmax=sample_rate // 2).T
    
    def transform(self, audio):
    
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap
    
        [f, t, x] = signal.spectral.spectrogram(
                        audio, 
                        window=ham_win,
                        nperseg=window_size, 
                        noverlap=overlap, 
                        detrend=False, 
                        return_onesided=True, 
                        mode='magnitude') 
        x = x.T
            
        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)
        
        return x


def calculate_logmel(audio_path, sample_rate, extractor):

    (audio, _) = read_audio(audio_path, target_fs=sample_rate)

    audio = audio / np.max(np.abs(audio))

    feature = extractor.transform(audio)

    return feature


def calculate_features(args):
    """Write features and infos of audios to a hdf5 file.
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    data_type = args.data_type
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    mel_bins = config.mel_bins

    corrupted_files = config.corrupted_files

    # Paths
    if data_type == 'development':
        audio_dir = os.path.join(dataset_dir, 'audio_train')
        meta_csv = os.path.join(args.dataset_dir, 'train.csv')
        
    elif data_type == 'test':
        audio_dir = os.path.join(dataset_dir, 'audio_test')
        meta_csv = os.path.join(args.dataset_dir, 'sample_submission.csv')

    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 'mini_{}.h5'.format(data_type))
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 '{}.h5'.format(data_type))
    
    create_folder(os.path.dirname(hdf5_path))

    # Load csv
    df = pd.DataFrame(pd.read_csv(meta_csv))

    audio_names = []
    
    if data_type == 'development':
        manually_verifications = []
        target_labels = []

    for row in df.iterrows():
        
        audio_name = row[1]['fname']
        audio_names.append(audio_name)
        
        if data_type == 'development':
            
            target_label = row[1]['label']
            manually_verification = row[1]['manually_verified']
        
            target_labels.append(target_label)
            manually_verifications.append(manually_verification)

    # Use partial data when set mini_data to True
    if mini_data:
        
        audios_num = 300
        random_state = np.random.RandomState(0)
        audio_indexes = np.arange(len(audio_names))
        random_state.shuffle(audio_indexes)
        audio_indexes = audio_indexes[0 : audios_num]
        
        audio_names = [audio_names[idx] for idx in audio_indexes]
        
        if data_type == 'development':
            
            target_labels = [target_labels[idx] for idx in audio_indexes]
            
            manually_verifications = [manually_verifications[idx] 
                                      for idx in audio_indexes]
        
        print("Number of audios: {}".format(len(audio_names)))

    # Feature extractor
    extractor = LogMelExtractor(sample_rate=sample_rate,
                                window_size=window_size,
                                overlap=overlap,
                                mel_bins=mel_bins)

    # Write out to h5 file
    hf = h5py.File(hdf5_path, 'w')

    hf.create_dataset(
        name='feature',
        shape=(0, mel_bins),
        maxshape=(None, mel_bins),
        dtype=np.float32)

    calculate_time = time.time()
    bgn_fin_indices = []

    # Extract feature for audios
    for (n, audio_name) in enumerate(audio_names):

        print(n, audio_name)

        # Extract feature
        if audio_name in corrupted_files:
            feature = np.zeros((0, mel_bins))
            
        else:
            audio_path = os.path.join(audio_dir, audio_name)
            feature = calculate_logmel(audio_path, sample_rate, extractor)
        
        print(feature.shape)

        # Write feature to hdf5
        bgn_indice = hf['feature'].shape[0]
        fin_indice = bgn_indice + feature.shape[0]

        hf['feature'].resize((fin_indice, mel_bins))
        hf['feature'][bgn_indice: fin_indice] = feature

        bgn_fin_indices.append((bgn_indice, fin_indice))

    # Write infos to hdf5
    hf.create_dataset(name='filename', 
                      data=[s.encode() for s in audio_names], 
                      dtype='S32')
    
    hf.create_dataset(name='bgn_fin_indices',
                      data=bgn_fin_indices,
                      dtype=np.int32)
    
    if data_type == 'development':
        
        hf.create_dataset(name='label', 
                          data=[s.encode() for s in target_labels], 
                          dtype='S32')
        
        hf.create_dataset(name='manually_verification',
                          data=manually_verifications,
                          dtype=np.int32)
        
    hf.close()
    
    print("Write out hdf5 file to {}".format(hdf5_path))
    print("Time spent: {} s".format(time.time() - calculate_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    subparsers = parser.add_subparsers(dest='mode')
    parser_logmel = subparsers.add_parser('logmel')
    parser_logmel.add_argument('--dataset_dir', type=str, required=True)
    parser_logmel.add_argument('--workspace', type=str, required=True)
    parser_logmel.add_argument('--data_type', type=str, choices=['development', 'test'], required=True)
    parser_logmel.add_argument('--mini_data', action='store_true', default=False)

    args = parser.parse_args()

    if args.mode == 'logmel':

        calculate_features(args)
