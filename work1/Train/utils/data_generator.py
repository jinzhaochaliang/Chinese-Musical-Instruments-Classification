import h5py
import numpy as np
import time
import pandas as pd
import logging
import matplotlib.pyplot as plt

from utilities import calculate_scalar, repeat_seq, scale
import config


class DataGenerator(object):
    
    def __init__(self, hdf5_path, batch_size, time_steps, 
        validation_csv=None, holdout_fold=None, seed=1234):
        """
        Inputs:
          hdf5_path: str, path of hdf5 data
          batch_size: int
          time_stes: int, number of frames of a logmel spectrogram patch
          validate_csv: string | None, if None then use all data for training
          holdout_fold: int
          seed: int, random seed
        """
        
        # Parameters
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)

        self.labels = config.labels
        lb_to_ix = config.lb_to_ix
        
        self.time_steps = time_steps
        self.hop_frames = self.time_steps // 2
        
        self.classes_num = len(self.labels)
        
        # Load data
        load_time = time.time()
        hf = h5py.File(hdf5_path, 'r')
            
        self.audio_names = np.array([s.decode() for s in hf['filename'][:]])
        self.x = hf['feature'][:]
        self.bgn_fin_indices = hf['bgn_fin_indices'][:]
        event_labels = hf['label'][:]
        self.y = np.array([lb_to_ix[s.decode()] for s in event_labels])
        self.manually_verifications = hf['manually_verification'][:]
        
        hf.close()
        
        logging.info('Loading data time: {:.3f} s'.format(
            time.time() - load_time))
        
        # Load validation
        if validation_csv:
            self.train_audio_indexes, self.validate_audio_indexes = \
                self.get_audio_indexes(validation_csv, holdout_fold)
                
        else:
            self.train_audio_indexes = np.arange(len(self.audio_names))
            self.validate_audio_indexes = np.array([])
                               
        logging.info('Training audios number: {}'.format(
            len(self.train_audio_indexes)))
            
        logging.info('Validation audios number: {}'.format(
            len(self.validate_audio_indexes)))
                    
        # calculate scalar
        (self.mean, self.std) = self.calculate_training_data_scalar()
    
        # Get training patches
        self.train_patch_bgn_fin_y_tuples = \
            self.calculate_patch_bgn_fin_y_tuples(self.train_audio_indexes)
        
        logging.info('Training patches number: {}'.format(
            len(self.train_patch_bgn_fin_y_tuples)))
    
    def get_audio_indexes(self, validation_csv, holdout_fold):
        """Get train and audio indexes from validation csv. 
        """
        
        df = pd.read_csv(validation_csv, sep=',')
        df = pd.DataFrame(df)
        
        folds = df['fold']
        
        train_audio_indexes = np.where(folds != holdout_fold)[0]
        validate_audio_indexes = np.where(folds == holdout_fold)[0]
        
        return train_audio_indexes, validate_audio_indexes

    def calculate_training_data_scalar(self):
        """Concatenate all training data and calculate scalar. 
        """
        
        train_bgn_fin_indices = self.bgn_fin_indices[self.train_audio_indexes]
        
        train_x_concat = []
        
        for [bgn, fin] in train_bgn_fin_indices:
            train_x_concat.append(self.x[bgn : fin])
            
        train_x_concat = np.concatenate(train_x_concat, axis=0)
        
        (mean, std) = calculate_scalar(train_x_concat)
        
        return mean, std
    
    def calculate_patch_bgn_fin_y_tuples(self, audio_indexes):
        """Calculate (bgn, fin, y) tuples for selecting patches for training. 
        """
        
        bgn_fin_indices = self.bgn_fin_indices[audio_indexes]
        
        patch_bgn_fin_y_tuples = []

        for n in range(len(audio_indexes)):
            
            [bgn, fin] = bgn_fin_indices[n]
            y = self.y[audio_indexes[n]]

            patch_tuples_for_this_audio = \
                self.get_patch_bgn_fin_y_tuples_for_an_audio(bgn, fin, y)
                    
            patch_bgn_fin_y_tuples += patch_tuples_for_this_audio
         
            
        # Print class wise number of patches
        patches_per_class = np.zeros(self.classes_num, dtype=np.int32)
        
        for k in range(self.classes_num):
            patches_per_class[k] = np.sum(
                [tuple[2] == k for tuple in patch_bgn_fin_y_tuples])
        
        if False:
            for k in range(self.classes_num):
                logging.info('{:<30}{}'.format(
                    self.labels[k], patches_per_class[k]))
        
        return patch_bgn_fin_y_tuples
        
    def get_patch_bgn_fin_y_tuples_for_an_audio(self, bgn, fin, y):
        """Get (bgn, fin, y) tuples in an audio. 
        """
        
        if fin - bgn <= self.time_steps:
            patch_tuples_for_this_audio = [(bgn, fin, y)]
            
        else:
            bgns = np.arange(bgn, fin - self.time_steps, self.hop_frames)
            patch_tuples_for_this_audio = []
            
            for bgn in bgns:
                patch_tuples_for_this_audio.append(
                    (bgn, bgn + self.time_steps, y))
                
        return patch_tuples_for_this_audio
    
    def generate_train(self):
        
        batch_size = self.batch_size
        patch_bgn_fin_y_tuples = self.train_patch_bgn_fin_y_tuples.copy()
        time_steps = self.time_steps

        patches_num = len(patch_bgn_fin_y_tuples)

        self.random_state.shuffle(patch_bgn_fin_y_tuples)

        iteration = 0
        pointer = 0

        while True:

            # Reset pointer
            if pointer >= patches_num:
                pointer = 0
                self.random_state.shuffle(patch_bgn_fin_y_tuples)

            # Get batch indexes
            batch_patch_bgn_fin_y_tuples = patch_bgn_fin_y_tuples[
                pointer: pointer + batch_size]
                
            pointer += batch_size

            iteration += 1
            
            (batch_x, batch_y) = self.get_batch_x_y(
                self.x, batch_patch_bgn_fin_y_tuples)
            
            # Transform data
            batch_x = self.transform(batch_x)

            yield batch_x, batch_y
        
    def get_batch_x_y(self, full_x, batch_patch_bgn_fin_y_tuples):
        """Get batch_x and batch_y, repeat is audio is short. 
        """
        
        batch_x = []
        batch_y = []
        
        for (bgn, fin, y) in batch_patch_bgn_fin_y_tuples:
            
            batch_y.append(y)
            
            if fin - bgn == self.time_steps:
                batch_x.append(full_x[bgn : fin])
                
            else:
                batch_x.append(repeat_seq(full_x[bgn : fin], self.time_steps))

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        
        return batch_x, batch_y

    def generate_validate_slices(self, data_type, manually_verified_only, 
                                 shuffle, max_audios_num=None):
        """Generate patches in an audio. 
        
        Args:
          data_type: 'train' | 'validate'
          manually_verified_only: bool
          shuffle: bool
          max_audios_num: int, set maximum audios to speed up validation
        """
        
        if data_type == 'train':
            audio_indexes = np.array(self.train_audio_indexes)
            
        elif data_type == 'validate':
            audio_indexes = np.array(self.validate_audio_indexes)
            
        else:
            raise Exception('Incorrect data_type!')
        
        if manually_verified_only:
            
            manually_verified_indexes = np.where(
                self.manually_verifications[audio_indexes]==1)[0]
                
            audio_indexes = audio_indexes[manually_verified_indexes]
        
        if shuffle:
            self.validate_random_state.shuffle(audio_indexes)
        
        for (n, audio_index) in enumerate(audio_indexes):
            
            if n == max_audios_num:
                break
            
            [bgn, fin] = self.bgn_fin_indices[audio_index]
            y = self.y[audio_index]
            audio_name = self.audio_names[audio_index]
            
            patch_tuples_for_this_audio = \
                self.get_patch_bgn_fin_y_tuples_for_an_audio(bgn, fin, y)
        
            (batch_x_for_an_audio, _) = self.get_batch_x_y(
                self.x, patch_tuples_for_this_audio)
        
            batch_x_for_an_audio = self.transform(batch_x_for_an_audio)

            yield batch_x_for_an_audio, y, audio_name
                
            
    def transform(self, x):
        """Transform data. 
        
        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)
          
        Returns:
          Transformed data. 
        """

        return scale(x, self.mean, self.std)
            
            
class TestDataGenerator(DataGenerator):
    
    def __init__(self, dev_hdf5_path, test_hdf5_path, time_steps, 
                 test_hop_frames):
        """Test data generator. 
        
        Args:
          dev_hdf5_path: str, path of development hdf5 file
          test_hdf5_path: str, path of test hdf5 file
          time_stes: int, number of frames of a logmel spectrogram patch
          test_hop_frames: int
        """
        
        super(TestDataGenerator, self).__init__(
            hdf5_path=dev_hdf5_path, 
            batch_size=None, 
            time_steps=time_steps,
            validation_csv=None)
        
        self.test_hop_frames = test_hop_frames
        
        self.corrupted_files = config.corrupted_files
        
        # Load test data
        load_time = time.time()
        hf = h5py.File(test_hdf5_path, 'r')

        self.test_audio_names = np.array([s.decode() for s in hf['filename'][:]])
        self.test_x = hf['feature'][:]
        self.test_bgn_fin_indices = hf['bgn_fin_indices'][:]
        
        hf.close()
        
        logging.info('Loading data time: {:.3f} s'.format(
            time.time() - load_time))
        
    def generate_test_slices(self):
        
        test_hop_frames = self.test_hop_frames
        corrupted_files = config.corrupted_files
        
        audio_indexes = range(len(self.test_audio_names))
        
        for (n, audio_index) in enumerate(audio_indexes):
            
            [bgn, fin] = self.test_bgn_fin_indices[audio_index]
            
            audio_name = self.test_audio_names[audio_index]
            
            if fin > bgn:
            
                patch_tuples_for_this_audio = \
                    self.get_patch_bgn_fin_y_tuples_for_an_audio(bgn, fin, y=None)
            
                (batch_x_for_an_audio, _) = \
                    self.get_batch_x_y(self.test_x, patch_tuples_for_this_audio)
            
                batch_x_for_an_audio = self.transform(batch_x_for_an_audio)
    
                yield batch_x_for_an_audio, audio_name