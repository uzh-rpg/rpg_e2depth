import os
import json
import logging
import argparse
import torch
from model.model import *
from model.loss import *
from model.metric import *
from torch.utils.data import DataLoader, ConcatDataset
from data_loader.dataset import *
from trainer import Trainer
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
from os.path import join
from logger import Logger

logging.basicConfig(level=logging.INFO, format='')


def concatenate_subfolders(base_folder, dataset_type, event_folder, depth_folder, transform=None):
    """
    Create an instance of ConcatDataset by aggregating all the datasets in a given folder
    """

    subfolders = os.listdir(base_folder)
    print('Found {} samples in {}'.format(len(subfolders), base_folder))
    train_datasets = []
    for dataset_name in subfolders:
        train_datasets.append(eval(dataset_type)(base_folder=join(base_folder, dataset_name),
                                                 event_folder=event_folder,
                                                 depth_folder=depth_folder,
                                                 transform=transform))
    concat_dataset = ConcatDataset(train_datasets)

    return concat_dataset


def main(config, resume, initial_checkpoint=None):
    train_logger = Logger()

    dataset_type, base_folder, event_folder, depth_folder = {}, {}, {}, {}

    # this will raise an exception is the env variable is not set
    preprocessed_datasets_folder = os.environ['PREPROCESSED_DATASETS_FOLDER']

    for split in ['train', 'validation']:
        dataset_type[split] = config['data_loader'][split]['type']
        base_folder[split] = join(preprocessed_datasets_folder, config['data_loader'][split]['base_folder'])
        event_folder[split] = config['data_loader'][split]['event_folder']
        depth_folder[split] = config['data_loader'][split]['depth_folder']

    train_dataset = concatenate_subfolders(base_folder['train'],
                                           dataset_type['train'],
                                           event_folder['train'],
                                           depth_folder['train'],
                                           transform=Compose([RandomRotationFlip(20.0, 0.5, 0.5),
                                                              RandomCrop(128)]))

    validation_dataset = concatenate_subfolders(base_folder['validation'],
                                                dataset_type['validation'],
                                                event_folder['validation'],
                                                depth_folder['validation'],
                                                transform=CenterCrop(128))

    # Set up data loaders
    kwargs = {'num_workers': config['data_loader']['num_workers'],
              'pin_memory': config['data_loader']['pin_memory']} if config['cuda'] else {}
    data_loader = DataLoader(train_dataset, batch_size=config['data_loader']['batch_size'],
                             shuffle=config['data_loader']['shuffle'], **kwargs)

    valid_data_loader = DataLoader(validation_dataset, batch_size=config['data_loader']['batch_size'],
                                   shuffle=config['data_loader']['shuffle'], **kwargs)

    # Set up model
    model = eval(config['arch'])(config['model'])

    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])

    # model.summary()

    loss = eval(config['loss']['type'])
    loss_params = config['loss']['config'] if 'config' in config['loss'] else None
    metrics = [eval(metric) for metric in config['metrics']]

    trainer = Trainer(model, loss, loss_params, metrics,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      train_logger=train_logger)

    trainer.train()

    path = os.path.join(config['trainer']['save_dir'], config['name'])
    with open(os.path.join(path, 'log.txt'), 'w') as f:
        f.write(train_logger.entries.__str__())


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Learning DVS Image Reconstruction')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-i', '--initial_checkpoint', default=None, type=str,
                        help='path to the checkpoint with which to initialize the model weights (default: None)')

    args = parser.parse_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        if args.initial_checkpoint is not None:
            logger.warning(
                'Warning: --initial_checkpoint overriden by --resume')
        config = torch.load(args.resume)['config']
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None

    main(config, args.resume, args.initial_checkpoint)
