# -*- coding: utf-8 -*-
"""
Dataset classes
"""

from torch.utils.data import Dataset
from .event_dataset import VoxelGridDataset
from skimage import io
from os.path import join
import numpy as np
from utils.util import first_element_greater_than, last_element_less_than
import random
import glob
import torch
import torch.nn.functional as f
from math import fabs


class SequenceSynchronizedFramesEventsDataset(Dataset):
    """Load sequences of time-synchronized {event tensors + depth} from a folder."""

    def __init__(self, base_folder, event_folder, depth_folder='frames', frame_folder='rgb', flow_folder='flow', semantic_folder='semantic',
                 start_time=0.0, stop_time=0.0,
                 sequence_length=2, transform=None,
                 proba_pause_when_running=0.0, proba_pause_when_paused=0.0,
                 step_size=20,
                 clip_distance=100.0,
                 normalize=True,
                 scale_factor = 1.0, inverse=False):
        assert(sequence_length > 0)
        assert(step_size > 0)
        assert(clip_distance > 0)
        self.L = sequence_length
        self.dataset = SynchronizedFramesEventsDataset(base_folder, event_folder, depth_folder, frame_folder, flow_folder, semantic_folder,
                                                       start_time, stop_time, clip_distance,
                                                       transform, normalize=normalize, inverse=inverse)
        self.event_dataset = self.dataset.event_dataset
        self.step_size = step_size
        if self.L >= self.dataset.length:
            self.length = 0
        else:
            self.length = (self.dataset.length - self.L) // self.step_size + 1

        self.proba_pause_when_running = proba_pause_when_running
        self.proba_pause_when_paused = proba_pause_when_paused
        self.scale_factor = scale_factor

    def __len__(self):
        return self.length

    def grid_sample_nans(self, y):
        B,_, H, W = y.shape
        D_H = torch.linspace(1, H, H)
        D_W = torch.linspace(1, W, W)
        meshy, meshx = torch.meshgrid((D_H, D_W))
        grid = torch.stack((meshy, meshx), 2)
        grid = grid.unsqueeze(0)
        return  torch.nn.functional.grid_sample(y, grid, padding_mode='border', align_corners=True)

    def __getitem__(self, i):
        """ Returns a list containing synchronized events <-> frame pairs
            [e_{i-L} <-> I_{i-L},
             e_{i-L+1} <-> I_{i-L+1},
            ...,
            e_{i-1} <-> I_{i-1},
            e_i <-> I_i]
        """
        assert(i >= 0)
        assert(i < self.length)

        # generate a random seed here, that we will pass to the transform function
        # of each item, to make sure all the items in the sequence are transformed
        # in the same way
        seed = random.randint(0, 2**32)

        # data augmentation: add random, virtual "pauses",
        # i.e. zero out random event tensors and repeat the last frame
        sequence = []

        # add the first element (i.e. do not start with a pause)
        k = 0
        j = i * self.step_size
        item = self.dataset.__getitem__(j, seed)
        sequence.append(item)

        paused = False
        for n in range(self.L - 1):

            # decide whether we should make a "pause" at this step
            # the probability of "pause" is conditioned on the previous state (to encourage long sequences)
            u = np.random.rand()
            if paused:
                probability_pause = self.proba_pause_when_paused
            else:
                probability_pause = self.proba_pause_when_running
            paused = (u < probability_pause)

            if paused:
                # add a tensor filled with zeros, paired with the last frame
                # do not increase the counter
                item = self.dataset.__getitem__(j + k, seed)
                item['events'].fill_(0.0)
                if 'flow' in item:
                    item['flow'].fill_(0.0)
                sequence.append(item)
            else:
                # normal case: append the next item to the list
                k += 1
                item = self.dataset.__getitem__(j + k, seed)
                sequence.append(item)

        # down sample data
        if self.scale_factor < 1.0:
            for data_items in sequence:
                for k, item in data_items.items():
                    if k is not "times":
                        item = item[None]
                        item = f.interpolate(item, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
                        item = item[0]
                        data_items[k] = item
        return sequence


class SynchronizedFramesEventsDataset(Dataset):
    """Loads time-synchronized event tensors and depth from a folder.

    This Dataset class iterates through all the event tensors and returns, for each tensor,
    a dictionary of the form:

        {'depth': frame, 'events': events, 'flow': disp_01, 'semantic': semantic}

    where:

    * depth is a H x W tensor containing the first frame whose timestamp >= event tensor
    * events is a C x H x W tensor containing the event data
    * flow is a 2 x H x W tensor containing the flow (displacement) from the current frame to the last frame
    * semantic is a 1 x H x W tensor containing the semantic labels 

    This loader assumes that each event tensor can be uniquely associated with a frame.
    For each event tensor with timestamp e_t, the corresponding frame is the first frame whose timestamp f_t >= e_t

    """

    def __init__(self, base_folder, event_folder, depth_folder='frames', frame_folder='rgb', flow_folder='flow', semantic_folder='semantic',
                 start_time=0.0, stop_time=0.0, clip_distance = 100.0,
                 transform=None, normalize=True, inverse=False):

        self.base_folder = base_folder
        self.event_folder = join(self.base_folder, event_folder if event_folder is not None else 'events')
        self.depth_folder = join(self.base_folder, depth_folder if depth_folder is not None else 'frames')
        self.frame_folder = join(self.base_folder, frame_folder if frame_folder is not None else 'rgb')
        self.flow_folder = join(self.base_folder, flow_folder if flow_folder is not None else 'flow')
        self.semantic_folder = join(self.base_folder, semantic_folder if semantic_folder is not None else 'semantic')
        self.transform = transform
        self.event_dataset = VoxelGridDataset(base_folder, event_folder,
                                              start_time, stop_time,
                                              transform=self.transform,
                                              normalize=normalize)
        self.eps = 1e-06
        self.clip_distance = clip_distance
        self.inverse = inverse

        # Load the stamp files
        self.stamps = np.loadtxt(
            join(self.depth_folder, 'timestamps.txt'))[:, 1]

        # shift the frame timestamps by the same amount as the event timestamps
        self.stamps -= self.event_dataset.initial_stamp

        self.length = len(self.event_dataset)

        # Check that the frame timestamps are unique and sorted
        assert(np.alltrue(np.diff(self.stamps) > 0)
               ), "frame timestamps are not unique and monotonically increasing"

        # Check that the latest frame in the dataset has a timestamp >= the latest event frame
        assert(
            self.stamps[-1] >= self.event_dataset.get_last_stamp())

    def __len__(self):
        return self.length

    def __getitem__(self, i, seed=None, reg_factor=3.70378):
        assert(i >= 0)
        assert(i < self.length)

        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

        def nan_helper(y):
            """Helper to handle indices and logical indices of NaNs.

            Input:
                - y, 1d numpy array with possible NaNs
            Output:
                - nans, logical indices of NaNs
                - index, a function, with signature indices= index(logical_indices),
                to convert logical indices of NaNs to 'equivalent' indices
            Example:
                >>> # linear interpolation of NaNs
                >>> nans, x= nan_helper(y)
                >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
            """
            return np.isnan(y), lambda z: z.nonzero()[0]

        event_timestamp = self.event_dataset.get_stamp_at(i)

        # Find the index of the first frame whose timestamp is >= event timestamp
        (frame_idx, frame_timestamp) = first_element_greater_than(
            self.stamps, event_timestamp)
        assert(frame_idx >= 0)
        assert(frame_idx < len(self.stamps))
        # assert(frame_timestamp >= event_timestamp)
        assert(frame_timestamp == event_timestamp)

        if seed is None:
            # if no specific random seed was passed, generate our own.
            # otherwise, use the seed that was passed to us
            seed = random.randint(0, 2**32)
 
        # tol = 0.01
        # if fabs(frame_timestamp - event_timestamp) > tol:
        #     print(
        #         'Warning: frame_timestamp and event_timestamp differ by more than tol ({} s)'.format(tol))
        #     print('frame_timestamp = {}, event_timestamp = {}'.format(
        #         frame_timestamp, event_timestamp))

        #frame = io.imread(join(self.depth_folder, 'frame_{:010d}.png'.format(frame_idx)),
        #                  as_gray=False).astype(np.float32) / 255.

        # Get the event tensor from the event dataset loader
        # Note that we pass the transform seed to ensure the same transform is used
        events = self.event_dataset.__getitem__(i, seed)

        # Load numpy depth ground truth frame 
        frame = np.load(join(self.depth_folder, 'depth_{:010d}.npy'.format(frame_idx))).astype(np.float32)

        #if np.isnan(frame).sum()>0:
            #events_mask = (torch.sum(events["events"], dim=0).unsqueeze(0))>0
            #frame[~events_mask] = 0
            # this (below) is the good one for mvsec
            #nans, x= nan_helper(frame)
            #frame[nans]= np.interp(x(nans), x(~nans), frame[~nans])

        # Clip to maximum distance
        frame = np.clip(frame, 0.0, self.clip_distance)
        # Normalize
        frame = frame / np.amax(frame[~np.isnan(frame)])
        #div = abs(np.min(np.log(frame+self.eps)))

        # Inverse depth
        if self.inverse:
            frame = 1.0 / frame
            frame = frame / np.amax(frame[~np.isnan(frame)])

        #Convert to log depth
        frame = 1.0 + np.log(frame) / reg_factor
        # Clip between 0 and 1.0
        frame = frame.clip(0, 1.0)

        if len(frame.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            frame = np.expand_dims(frame, -1)

        frame = np.moveaxis(frame, -1, 0)  # H x W x C -> C x H x W
        frame = torch.from_numpy(frame) #numpy to tensor

        # Get RGB frames
        if self.frame_folder is not None:
            try:
                rgb_frame = io.imread(join(self.frame_folder, 'frame_{:010d}.png'.format(frame_idx)), as_gray=False).astype(np.float32)
                if rgb_frame.shape[2] > 1:
                    gray_frame = rgb2gray(rgb_frame) #[H x W]
                
                gray_frame /= 255.0 #normalize
                gray_frame = np.expand_dims(gray_frame, axis=0) #expand to [1 x H x W]
                gray_frame = torch.from_numpy(gray_frame) #numpy to tensor
                if self.transform:
                    random.seed(seed)
                    gray_frame = self.transform(gray_frame)

                # Combine events with grayscale frames
                #events["events"] = torch.cat((events["events"], gray_frame), axis=0)
                events["events"] = gray_frame
            except FileNotFoundError:
                gray_frame = None

        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)

        # Get optic flow tensor and apply the same transformation as the others
        if self.flow_folder is not None:
            try:
                flow = np.load(join(self.flow_folder, 'disp01_{:010d}.npy'.format(i + 1))).astype(np.float32)
                flow = torch.from_numpy(flow)  # [2 x H x W]
                if self.transform:
                    random.seed(seed)
                    flow = self.transform(flow, is_flow=True)
            except FileNotFoundError:
                flow = None

        # Get the semantic label tensor and apply the same transformation as the others
        if self.semantic_folder is not None:
            try:
                semantic = np.load(join(self.semantic_folder, 'semantic_{:010d}.npy'.format(i + 1))).astype(np.float32)
                semantic = torch.from_numpy(semantic)  # [1 x H x W]
                if self.transform:
                    random.seed(seed)
                    semantic = self.transform(semantic)
            except FileNotFoundError:
                semantic = None


        # Merge the 'frame' dictionary with the 'events' one
        if flow is not None and semantic is not None:
            item = {'frame': frame,
                    'flow': flow,
                    'semantic': semantic,
                    **events}
        else:
            if flow is not None:
                item = {'frame': frame,
                        'flow': flow,
                        **events}
            elif semantic is not None:
                item = {'frame': frame,
                        'semantic': semantic,
                        **events}
            else:
                item = {'frame': frame,
                        **events}
            return item


class EventsBetweenFramesDataset(Dataset):
    """Loads time-synchronized event tensors and frame-pairs from a folder.

    This Dataset class iterates through all the event tensors and returns, for each tensor,
    a dictionary of the form:

        {'depth': [frame0, frame1], 'events': events}

    where:

    * depth is a tuple containing two H x W tensor containing the start/end frames
    * events is a C x H x W tensor containing the events in that were triggered in between the frames

    This loader assumes that each event tensor can be uniquely associated with a frame pair.
    For each event tensor with timestamp e_t, the corresponding frame pair is [frame_idx-1, frame_idx], where
    frame_idx is the index of the first frame whose timestamp f_t >= e_t

    """

    def __init__(self, base_folder, event_folder, depth_folder='frames',
                 start_time=0.0, stop_time=0.0, transform=None, normalize=True):
        self.base_folder = base_folder
        self.depth_folder = join(self.base_folder, depth_folder if depth_folder is not None else 'frames')
        self.transform = transform

        self.event_dataset = VoxelGridDataset(base_folder, event_folder,
                                              start_time, stop_time,
                                              transform=self.transform,
                                              normalize=normalize)

        # Load the frame stamps file
        self.stamps = np.loadtxt(
            join(self.depth_folder, 'timestamps.txt'))[:, 1]

        # Shift the frame timestamps by the same amount as the event timestamps
        self.stamps -= self.event_dataset.initial_stamp

        self.length = len(self.event_dataset)

        # Check that the frame timestamps are unique and sorted
        assert(np.alltrue(np.diff(self.stamps) > 0)
               ), "frame timestamps are not unique and monotonically increasing"

        # Load the event boundaries stamps file
        # (it is a file containing the index of the first/last event for every event tensor)
        self.boundary_stamps = np.loadtxt(
            join(self.event_dataset.event_folder, 'boundary_timestamps.txt'))[:, 1:]

        # Shift the boundary timestamps by the same amount as the event timestamps
        self.boundary_stamps[:, 0] -= self.event_dataset.initial_stamp
        self.boundary_stamps[:, 1] -= self.event_dataset.initial_stamp

        # Check the the first event timestamp >= the first frame in the dataset
        assert(
            self.stamps[0] <= self.boundary_stamps[self.event_dataset.first_valid_idx, 0])

        # Check that the latest frame in the dataset has a timestamp >= the latest event frame
        assert(
            self.stamps[-1] >= self.boundary_stamps[self.event_dataset.last_valid_idx, 1])

    def __len__(self):
        return self.length

    def __getitem__(self, i, seed=None):
        assert(i >= 0)
        assert(i < self.length)

        # stamp of the first event in the tensor
        et0 = self.boundary_stamps[self.event_dataset.get_index_at(i), 0]
        # stamp of the last event in the tensor
        et1 = self.boundary_stamps[self.event_dataset.get_index_at(i), 1]

        # print('i = ', i)
        # print('et0, et1 = ', et0 + self.event_dataset.initial_stamp,
        #       et1 + self.event_dataset.initial_stamp)

        # Find the index of the last frame whose timestamp is <= et0
        (frame0_idx, frame0_timestamp) = last_element_less_than(
            self.stamps, et0)
        assert(frame0_idx >= 0)
        assert(frame0_idx < len(self.stamps))
        assert(frame0_timestamp <= et0)

        tol = 0.01
        if fabs(frame0_timestamp - et0) > tol:
            print(
                'Warning: frame0_timestamp and et0 differ by more than tol ({} s)'.format(tol))
            print('frame0_timestamp = {}, et0 = {}'.format(
                frame0_timestamp, et0))

        frame0 = io.imread(join(self.depth_folder, 'frame_{:010d}.png'.format(frame0_idx)),
                           as_gray=False).astype(np.float32) / 255.

        if len(frame0.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            frame0 = np.expand_dims(frame0, -1)

        frame0 = np.moveaxis(frame0, -1, 0)  # H x W x C -> C x H x W
        frame0 = torch.from_numpy(frame0)

        # Find the index of the first frame whose timestamp is >= et1
        (frame1_idx, frame1_timestamp) = first_element_greater_than(
            self.stamps, et1)
        assert(frame1_idx >= 0)
        assert(frame1_idx < len(self.stamps))
        assert(frame1_timestamp >= et1)

        if fabs(frame1_timestamp - et1) > tol:
            print(
                'Warning: frame1_timestamp and et1 differ by more than tol ({} s)'.format(tol))
            print('frame1_timestamp = {}, et1 = {}'.format(
                frame1_timestamp, et1))

        frame1 = io.imread(join(self.depth_folder, 'frame_{:010d}.png'.format(frame1_idx)),
                           as_gray=False).astype(np.float32) / 255.
        frame1 = torch.from_numpy(frame1).unsqueeze(dim=0)  # [H x W] -> [1 x H x W]

        if len(frame1.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            frame1 = np.expand_dims(frame1, -1)

        frame1 = np.moveaxis(frame1, -1, 0)  # H x W x C -> C x H x W
        frame1 = torch.from_numpy(frame1)

        # print('ft0, ft1 = ', frame0_timestamp + self.event_dataset.initial_stamp,
        #       frame1_timestamp + self.event_dataset.initial_stamp)
        # print('f_idx0, f_idx1 = ', frame0_idx, frame1_idx)

        if seed is None:
            # if no specific random seed was passed, generate our own.
            # otherwise, use the seed that was passed to us
            seed = random.randint(0, 2**32)

        if self.transform:
            random.seed(seed)
            frame0 = self.transform(frame0)
            random.seed(seed)
            frame1 = self.transform(frame1)

        # Get the event tensor from the event dataset loader
        # Note that we pass the transform seed to ensure the same transform is used
        events = self.event_dataset.__getitem__(i, seed)

        # Merge the 'frame' dictionary with the 'events' one
        item = {'frames': [frame0, frame1],
                **events}

        return item
