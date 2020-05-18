import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import time
import argparse
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math
import h5py
import sys
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from utilities import (create_folder, get_filename, create_logging, 
                       calculate_accuracy, 
                       print_class_wise_accuracy, plot_class_wise_accuracy, 
                       write_testing_data_submission_csv)
from average_precision import mapk
from models_pytorch import move_data_to_gpu, BaselineCnn, Vggish, CNN8_Cent, Vggish4,Vggish6
from data_generator import DataGenerator, TestDataGenerator
import config

from tensorboardX import SummaryWriter


# Hyper-parameters
Model = Vggish
# Model = BaselineCnn
# Model = CNN8_Cent
batch_size = 64
time_steps = 128
train_hop_frames = 64
test_hop_frames = 16
kmax = config.kmax

writer = SummaryWriter()
       
def aggregate_outputs(outputs):
    """Aggregate the prediction of patches of audio clips. 
    
    Args:
      outputs: (audios_num, patches_num, classes_num)
      
    Returns:
      agg_outputs: (audios_num)
    """
    
    agg_outputs = []
    
    for output in outputs:
        agg_output = np.mean(output, axis=0)
        agg_outputs.append(agg_output)
    
    agg_outputs = np.array(agg_outputs)
    
    return agg_outputs
    
def evaluate(model, generator, data_type, cuda):
    """Evaluate
    
    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      cuda: bool.
      
    Returns:
      accuracy: float
      mapk: float
    """

    if data_type == 'train':
        max_audios_num = 15000   # A small portion of training data to evaluate
    
    elif data_type == 'validate':
        max_audios_num = None   # All evaluation data to evaluate

    generate_func = generator.generate_validate_slices(
        data_type=data_type, 
        manually_verified_only=True, 
        shuffle=True, 
        max_audios_num=max_audios_num)
    
    # Forward
    dict = forward(model=model, 
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_target=True)

    outputs = dict['output']    # (audios_num, patches_num, classes_num)
    targets = dict['target']    # (audios_num,)

    agg_outputs = aggregate_outputs(outputs)
    '''(audios_num, classes_num)'''

    predictions = np.argmax(agg_outputs, axis=-1)
    '''(audios_num,)'''
    
    sorted_indices = np.argsort(agg_outputs, axis=-1)[:, ::-1][:, :kmax]
    '''(audios_num, kmax)'''

    # Accuracy
    accuracy = calculate_accuracy(predictions, targets)

    # mAP
    mapk_value = mapk(actual=[[e] for e in targets], 
                      predicted=[e.tolist() for e in sorted_indices], 
                      k=kmax)

    return accuracy, mapk_value
    
    
def forward(model, generate_func, cuda, return_target):
    """Forward data to a model.
    
    Args:
      generate_func: generate function
      cuda: bool
      return_target: bool
      
    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """
    
    outputs = []
    audio_names = []
    
    if return_target:
        targets = []
    
    # Evaluate on mini-batch
    for data in generate_func:
            
        if return_target:
            (batch_x_for_an_audio, y, audio_name) = data
            
        else:
            (batch_x_for_an_audio, audio_name) = data
            
        batch_x_for_an_audio = move_data_to_gpu(batch_x_for_an_audio, cuda)

        # Predict
        model.eval()
        outputs_for_an_audio = model(batch_x_for_an_audio)

        # logging.info('{}'.format(outputs_for_an_audio.data.cpu().numpy().shape))
        # logging.info('{}'.format(audio_name))
        # Append data
        outputs.append(outputs_for_an_audio.data.cpu().numpy())
        audio_names.append(audio_name)
        
        if return_target:
            targets.append(y)

    dict = {}

    outputs = np.array(outputs)
    dict['output'] = outputs
    
    audio_names = np.array(audio_names)
    dict['audio_name'] = audio_names
    
    if return_target:
        targets = np.array(targets)
        dict['target'] = targets
        
    return dict
    
    
def train(args):
    
    # Arguments & parameters
    workspace = args.workspace
    filename = args.filename
    validate = args.validate
    holdout_fold = args.holdout_fold
    cuda = args.cuda
    mini_data = args.mini_data
    
    num_classes = len(config.labels)

    # Use partial data for training
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel',
                                 'mini_development.h5')        
        
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel',
                                 'development.h5')
    
    if validate:
        validation_csv = os.path.join(workspace, 'validate_meta.csv')
        
        models_dir = os.path.join(workspace, 'models', filename, 
            'holdout_fold{}'.format(holdout_fold))
        
    else:
        validation_csv = None
        
        models_dir = os.path.join(workspace, 'models', filename, 'full_train')
    
    create_folder(models_dir)
    
    # Model
    model = Model(num_classes)
    
    if cuda:
        model.cuda()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), 
                           eps=1e-08, weight_decay=0.)
    
    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path, 
                              batch_size=batch_size, 
                              time_steps=time_steps, 
                              validation_csv=validation_csv, 
                              holdout_fold=holdout_fold)

    iteration = 0
    train_bgn_time = time.time()

    # Train on mini batches
    for (batch_x, batch_y) in generator.generate_train():
        
        # Evaluate
        if iteration % 100 == 0:

            train_fin_time = time.time()
            
            (tr_acc, tr_mapk) = evaluate(model=model, 
                                         generator=generator, 
                                         data_type='train', 
                                         cuda=cuda)
                                         
            logging.info('train acc: {:.6f}, train mapk: {:.6f}'.format(
                tr_acc, tr_mapk))
            
            writer.add_scalars('scalars/train',{'train_acc':tr_acc,'train_mapk':tr_mapk},iteration)
                        
            if validate:
                (va_acc, va_mapk) = evaluate(model=model, 
                                             generator=generator, 
                                             data_type='validate', 
                                             cuda=cuda)
                                             
                logging.info('valid acc: {:.3f}, validate mapk: {:.3f}'.format(
                    va_acc, va_mapk))
        
            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            
            logging.info('------------------------------------')
            logging.info('Iteration: {}, train time: {:.3f} s, eval time: '
                '{:.3f} s'.format(iteration, train_time, validate_time))
            
            train_bgn_time = time.time()
            
        # Save model
        if iteration % 1000 == 0 and iteration > 0:
            save_out_dict = {'iteration': iteration, 
                             'state_dict': model.state_dict(), 
                             'optimizer': optimizer.state_dict(), }
            
            save_out_path = os.path.join(models_dir, 
                'md_{}_iters.tar'.format(iteration))
                
            torch.save(save_out_dict, save_out_path)
            logging.info('Save model to {}'.format(save_out_path))
            
        # Reduce learning rate
        if iteration % 100 == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
        
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        
        # Forward
        t_forward = time.time()
        model.train()
        output = model(batch_x)
        
        # Loss
        loss = F.nll_loss(output, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('scalar/loss',loss.data.item(),iteration)
        
        iteration += 1
        
        if iteration%200 == 0:
            logging.info('iteration: {}, loss: {:.6}'.format(iteration, loss.data.item()))
        
        # Stop learning
        if iteration == 5001:
            writer.close()
            break
    
    
def inference_validation_data(args):
    
    # Arguments & parameters
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda
    
    classes_num = len(config.labels)
    
    # Paths    
    model_path = os.path.join(workspace, 'models', filename, 
                              'holdout_fold{}'.format(holdout_fold), 
                              'md_{}_iters.tar'.format(iteration))
    
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                             'development.h5')
    
    validation_csv = os.path.join(workspace, 'validate_meta.csv')
    
    stats_pickle_path = os.path.join(workspace, 'stats', filename, 
                                     'holdout_fold{}'.format(holdout_fold), 
                                     '{}_iters.p'.format(iteration))
    
    create_folder(os.path.dirname(stats_pickle_path))
    
    # Model
    model = Model(classes_num)
        
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if cuda:
        model.cuda()
    
    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path, 
                              batch_size=batch_size, 
                              time_steps=time_steps, 
                              validation_csv=validation_csv, 
                              holdout_fold=holdout_fold)
    
    generate_func = generator.generate_validate_slices(
        data_type='validate', 
        manually_verified_only=True, 
        shuffle=False, 
        max_audios_num=None)
    
    # Forward
    dict = forward(model=model, 
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_target=True)

    outputs = dict['output']    # (audios_num, patches_num, classes_num)
    targets = dict['target']    # (audios_num,)

    agg_outputs = aggregate_outputs(outputs)
    '''(audios_num, classes_num)'''

    predictions = np.argmax(agg_outputs, axis=-1)
    '''(audios_num,)'''
    
    sorted_indices = np.argsort(agg_outputs, axis=-1)[:, ::-1][:, :kmax]
    '''(audios_num, kmax)'''

    # Accuracy
    accuracy = calculate_accuracy(predictions, targets)

    # mAP
    mapk_value = mapk(actual=[[e] for e in targets], 
                      predicted=[e.tolist() for e in sorted_indices], 
                      k=kmax)
    
    # Print
    logging.info('')
    logging.info('iteration: {}'.format(iteration))
    logging.info('accuracy: {:.3f}'.format(accuracy))
    logging.info('mapk: {:.3f}'.format(mapk_value))
    
    (class_wise_accuracy, correctness, total) = print_class_wise_accuracy(
        predictions, targets)
        
    # Save stats for current holdout training
    dict = {'correctness': correctness, 'total': total, 
            'accuracy': accuracy, 'mapk': mapk_value}
    
    pickle.dump(dict, open(stats_pickle_path, 'wb'))
    
    logging.info('Write out stat to {}'.format(stats_pickle_path))
    

def inference_testing_data(args):
    
    # Arguments & parameters
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda
    
    num_classes = len(config.labels)
    
    # Paths
    model_path = os.path.join(workspace, 'models', filename, 'full_train', 
                              'md_{}_iters.tar'.format(iteration))
    
    dev_hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 'development.h5')
                                 
    test_hdf5_path = os.path.join(workspace, 'features', 'logmel', 'test.h5')
    
    submission_path = os.path.join(workspace, 'submissions', filename, 
                                   'iteration={}'.format(iteration), 
                                   'submission.csv')
    
    create_folder(os.path.dirname(submission_path))
    
    # Model
    model = Model(num_classes)
        
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if cuda:
        model.cuda()
    
    # Data generator
    test_generator = TestDataGenerator(dev_hdf5_path=dev_hdf5_path, 
                                       test_hdf5_path=test_hdf5_path, 
                                       test_hop_frames=test_hop_frames, 
                                       time_steps=time_steps)
    
    generate_func = test_generator.generate_test_slices()
    
    # Forward
    dict = forward(model=model, 
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_target=False)
    
    outputs = dict['output']
    audio_names = dict['audio_name']
    
    agg_outputs = aggregate_outputs(outputs)
    '''(audios_num, classes_num)'''

    predictions = np.argmax(agg_outputs, axis=-1)
    '''(audios_num,)'''
    
    sorted_indices = np.argsort(agg_outputs, axis=-1)[:, ::-1][:, :kmax]
    '''(audios_num, kmax)'''
    
    # Write out submission csv
    write_testing_data_submission_csv(submission_path, audio_names, 
                                      sorted_indices)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--holdout_fold', type=int, choices=[1, 2, 3, 4])
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    
    parser_inference_validation_data = subparsers.add_parser('inference_validation_data')
    parser_inference_validation_data.add_argument('--workspace', type=str, required=True)    
    parser_inference_validation_data.add_argument('--holdout_fold', type=int, choices=[1, 2, 3, 4])
    parser_inference_validation_data.add_argument('--iteration', type=str, required=True)
    parser_inference_validation_data.add_argument('--cuda', action='store_true', default=False)
    
    parser_inference_testing_data = subparsers.add_parser('inference_testing_data')
    parser_inference_testing_data.add_argument('--workspace', type=str, required=True)    
    parser_inference_testing_data.add_argument('--verified_only', action='store_true', default=False)
    parser_inference_testing_data.add_argument('--iteration', type=str, required=True)
    parser_inference_testing_data.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    
    args.filename = get_filename(__file__)
    
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)    
    logging = create_logging(logs_dir, filemode='w')
    logging.info(args)
    
    if args.mode == 'train':          
        train(args)
        
    elif args.mode == 'inference_validation_data':
        inference_validation_data(args)

    elif args.mode == 'inference_testing_data':
        inference_testing_data(args)
        
    else:
        raise Exception('Error!')