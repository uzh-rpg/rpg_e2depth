import torch
from utils.loading_utils import load_model, get_device
from utils.event_readers import VoxelGridDataset 
from os.path import join, basename
import numpy as np
import json
import argparse
import shutil
import os
from depth_prediction import DepthEstimator
from options.inference_options import set_depth_inference_options

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                        help='path to model weights')
    parser.add_argument('-i', '--input_folder', default=None, type=str,
                        help="name of the folder containing the voxel grids")
    parser.add_argument('--start_time', default=0.0, type=float)
    parser.add_argument('--stop_time', default=0.0, type=float)

    set_depth_inference_options(parser)

    args = parser.parse_args()

    print_every_n = 50

    # Load model to device
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)
    model = model.to(device)
    model.eval()

    base_folder = os.path.dirname(args.input_folder)
    event_folder = os.path.basename(args.input_folder)

    # hack to get the image size: create a dummy dataset,
    # grab the first data item and read the required info
    dummy_dataset = VoxelGridDataset(base_folder,
                                     event_folder,
                                     args.start_time,
                                     args.stop_time,
                                     transform=None)
    data = dummy_dataset[0]
    _, height, width = data['events'].shape
    height = height - args.low_border_crop
    
    estimator = DepthEstimator(model, height, width, model.num_bins, args)

    events_dataset = VoxelGridDataset(base_folder=base_folder,
                               event_folder = event_folder,
                               start_time=args.start_time,
                               stop_time=args.stop_time,
                               transform=None)

    output_dir = args.output_folder
    dataset_name = args.dataset_name
    print('Processing {}'.format(dataset_name), end=': ')

    N = len(events_dataset)

    if output_dir is not None:
        shutil.copyfile(join(args.input_folder, 'timestamps.txt'),
                        join(output_dir, dataset_name, 'timestamps.txt'))
        shutil.copyfile(join(args.input_folder, 'boundary_timestamps.txt'),
                        join(output_dir, dataset_name, 'boundary_timestamps.txt'))

    idx = 0
    while idx < N:
        if idx % print_every_n == 0:
            print('{} / {}'.format(idx, N))

        data = events_dataset[idx]
        event_tensor = data['events'][:,:height,:]

        estimator.update_reconstruction(event_tensor, idx)
        idx += 1
