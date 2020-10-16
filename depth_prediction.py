import torch
import cv2
import numpy as np
from model.model import *
from utils.inference_utils import CropParameters, ImageDepthWriter, DepthDisplay
from utils.event_tensor_utils import EventPreprocessor
from utils.util import robust_min, robust_max
from utils.timers import CudaTimer, cuda_timers
from os.path import join
from collections import deque


class DepthEstimator:
    def __init__(self, model, height, width, num_bins, options):

        self.model = model
        self.use_gpu = options.use_gpu
        self.device = torch.device('cuda:0') if self.use_gpu else torch.device('cpu')
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.options = options

        self.initialize(self.height, self.width, self.options)

    def initialize(self, height, width, options):
        print('== Image reconstruction == ')
        print('Image size: {}x{}'.format(self.height, self.width))

        self.last_stamp = None

        self.no_recurrent = options.no_recurrent
        if self.no_recurrent:
            print('!!Recurrent connection disabled!!')

        self.crop = CropParameters(self.width, self.height, self.model.num_encoders)

        self.last_states_for_each_channel = {'grayscale': None}

        self.event_preprocessor = EventPreprocessor(options)
        self.image_writer = ImageDepthWriter(options)
        self.image_display = DepthDisplay(options)

    def update_reconstruction(self, event_tensor, event_tensor_id, stamp=None):

        # max duration without events before we reinitialize
        self.max_duration_before_reinit_s = 5.0

        # we reinitialize if stamp < last_stamp, or if stamp > last_stamp + max_duration_before_reinit_s
        if stamp is not None and self.last_stamp is not None:
            if stamp < self.last_stamp or stamp > self.last_stamp + self.max_duration_before_reinit_s:
                print('Reinitialization detected!')
                self.initialize(self.height, self.width, self.options)

        self.last_stamp = stamp

        with torch.no_grad():

            with CudaTimer('Reconstruction'):

                with CudaTimer('NumPy (CPU) -> Tensor (GPU)'):
                    events = event_tensor.unsqueeze(dim=0)
                    events = events.to(self.device)

                if self.options.use_fp16:
                    events = events.half()

                events = self.event_preprocessor(events)

                # Resize tensor to [1 x C x crop_size x crop_size] by applying zero padding
                events_for_each_channel = {'grayscale': self.crop.pad(events)}
                reconstructions_for_each_channel = {}

                # Reconstruct new intensity image for each channel (grayscale )
                for channel in events_for_each_channel.keys():
                    with CudaTimer('Inference'):
                        new_predicted_frame, states = self.model(events_for_each_channel[channel],
                                                                 self.last_states_for_each_channel[channel])

                    if self.no_recurrent:
                        self.last_states_for_each_channel[channel] = None
                    else:
                        self.last_states_for_each_channel[channel] = states

                    # Output reconstructed image
                    crop = self.crop if channel == 'grayscale' else self.crop_halfres

                    # Intensity rescaler (on GPU)
                    #new_predicted_frame = self.intensity_rescaler(new_predicted_frame)

                    with CudaTimer('Tensor (GPU) -> NumPy (CPU)'):
                        reconstructions_for_each_channel[channel] = new_predicted_frame[0, 0, crop.iy0:crop.iy1,
                                                                                        crop.ix0:crop.ix1].cpu().numpy()

                out = reconstructions_for_each_channel['grayscale']
                #print ("out", out)

            self.image_writer(out, event_tensor_id, stamp, events=events)
            self.image_display(out, events)
