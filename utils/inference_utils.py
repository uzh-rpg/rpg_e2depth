from .util import robust_min, robust_max, ensure_dir
from .timers import Timer
from .loading_utils import get_device
from os.path import join
import os
from math import ceil, floor
from torch.nn import ReflectionPad2d
import numpy as np
import torch
import cv2
from collections import deque
import atexit
import torch.nn.functional as F
from math import sqrt
import matplotlib as mpl
import matplotlib.cm as cm

def make_event_preview(events, mode='red-blue', num_bins_to_show=-1):
    # events: [1 x C x H x W] event tensor
    # mode: 'red-blue' or 'grayscale'
    # num_bins_to_show: number of bins of the voxel grid to show. -1 means show all bins.
    assert(mode in ['red-blue', 'grayscale'])
    if num_bins_to_show < 0:
        sum_events = torch.sum(events[0, :, :, :], dim=0).detach().cpu().numpy()
    else:
        sum_events = torch.sum(events[0, -num_bins_to_show:, :, :], dim=0).detach().cpu().numpy()

    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.zeros((sum_events.shape[0], sum_events.shape[1], 3), dtype=np.uint8)
        b = event_preview[:, :, 0]
        r = event_preview[:, :, 2]
        b[sum_events > 0] = 255
        r[sum_events < 0] = 255
    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        m, M = -10.0, 10.0
        event_preview = np.clip((255.0 * (sum_events - m) / (M - m)).astype(np.uint8), 0, 255)

    return event_preview


class ImageDepthWriter:
    """
    Utility class to write images to disk.
    Also writes the image timestamps into a text file.
    """

    def __init__(self, options):

        self.output_folder = options.output_folder
        self.dataset_name = options.dataset_name
        self.save_events = options.show_events
        self.save_numpy = options.save_numpy
        self.event_display_mode = options.event_display_mode
        self.num_bins_to_show = options.num_bins_to_show
        self.save_inv_log = options.save_inv_log
        self.save_inv = options.save_inv
        self.save_color_map = options.save_color_map

        print('== Image Writer ==')
        if self.output_folder:
            ensure_dir(self.output_folder)
            ensure_dir(join(self.output_folder, self.dataset_name))
            self.frames_dir = join(self.output_folder, self.dataset_name, "frames")
            ensure_dir(self.frames_dir)
            print('Will write images to: {}'.format(self.frames_dir))
            self.timestamps_file = open(join(self.frames_dir, 'timestamps.txt'), 'a')

            if self.save_events:
                self.event_previews_folder = join(self.output_folder, self.dataset_name, 'events')
                ensure_dir(self.event_previews_folder)
                print('Will write event previews to: {}'.format(self.event_previews_folder))

            atexit.register(self.__cleanup__)
            if self.save_numpy:
                self.data_folder = join(self.output_folder, self.dataset_name, 'data')
                ensure_dir(self.data_folder)
                print('Will write depth in numpy format into: {}'.format(self.data_folder))
        else:
            print('Will not write images to disk.')

    def __call__(self, img, event_tensor_id, stamp=None, events=None, reg_factor=3.70378):
        if not self.output_folder:
            return

        if self.save_events and events is not None:
            event_preview = make_event_preview(events, mode=self.event_display_mode,
                                               num_bins_to_show=self.num_bins_to_show)
            cv2.imwrite(join(self.event_previews_folder,
                             'events_{:010d}.png'.format(event_tensor_id)), event_preview)
        if self.save_numpy:
            # Write prediction as it comes
            data = img
            np.save(join(self.data_folder, 'depth_{:010d}.npy'.format(event_tensor_id)), data)

        if self.save_inv_log:
            # Convert to normalized depth
            img = np.exp(reg_factor * (img - np.ones((img.shape[0], img.shape[1]), dtype=np.float32)))
            # Perform inverse depth
            img = 1/img
            img = img/np.amax(img)
            #Convert back to log depth (it is now log inverse depth)
            img = np.ones((img.shape[0], img.shape[1]), dtype=np.float32) + np.log(img)/reg_factor
        elif self.save_inv:
            # Convert to normalized depth
            img = np.exp(reg_factor * (img - np.ones((img.shape[0], img.shape[1]), dtype=np.float32)))
            # Perform inverse depth
            img = 1/img
            img = img/np.amax(img)

        if self.save_color_map:
            vmax = np.percentile(img, 95)
            normalizer = mpl.colors.Normalize(vmin=img.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            img = mapper.to_rgba(img)
            img[:,:,0:3] = img[:,:,0:3][...,::-1]

        cv2.imwrite(join(self.frames_dir, 'frame_{:010d}.png'.format(event_tensor_id)), img*255.0)
        if stamp is not None:
            self.timestamps_file.write('{:.18f}\n'.format(stamp))

    def __cleanup__(self):
        if self.output_folder:
            self.timestamps_file.close()

def optimal_crop_size(max_size, max_subsample_factor):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    return crop_size

class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ReflectionPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)

class DepthDisplay:
    """
    Utility class to display depth reconstructions
    """

    def __init__(self, options):
        self.display = options.display
        self.display_trackbars = not options.no_display_trackbars
        self.show_reconstruction = not options.no_show_reconstruction
        self.show_events = options.show_events
        self.event_display_mode = options.event_display_mode
        self.num_bins_to_show = options.num_bins_to_show

        self.window_name = 'EventsDepth'

        self.inv_depth = 0
        self.log_depth = 1
        self.color = 0

        if self.display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

            if self.display_trackbars:
                # create switch for ON/OFF functionality
                switch = 'INV'
                cv2.createTrackbar(switch, self.window_name,0,1,self.on_inv_depth)

                # create switch for ON/OFF functionality
                switch = 'LOG'
                cv2.createTrackbar(switch, self.window_name,1,1,self.on_log_depth)

                # create switch for ON/OFF functionality
                switch = 'COLOR'
                cv2.createTrackbar(switch, self.window_name,0,1,self.on_color)

        self.border = options.display_border_crop
        self.wait_time = options.display_wait_time
    
    def on_inv_depth(self, tick_pos):
        self.inv_depth = tick_pos

    def on_log_depth(self, tick_pos):
        self.log_depth = tick_pos

    def on_color(self, tick_pos):
        self.color = tick_pos

    def crop_outer_border(self, img, border):
        if self.border == 0:
            return img
        else:
            return img[border:-border, border:-border]

    def __call__(self, img, events=None, reg_factor=5.70378):

        if not self.display:
            return

        img = self.crop_outer_border(img, self.border)
        
        with Timer('Inv Depth'):
            if self.inv_depth is 1:
                # Check if the image is in log depth
                if self.log_depth is 1:
                    # Convert to depth
                    img = np.exp(reg_factor * (img - np.ones((img.shape[0], img.shape[1]), dtype=np.float32)))
                    # Perform inverse depth
                    img = 1/img
                    img = img/np.amax(img)
                    #Convert back to log depth
                    img = np.ones((img.shape[0], img.shape[1]), dtype=np.float32) + np.log(img)/reg_factor
                else:
                    # Perform inverse depth
                    img = 1/img
                    img = img/np.amax(img)

        with Timer('Log Depth'):
            if self.log_depth is 0:
                if self.inv_depth is 0:
                    # Perform norm depth (the prediction is already in log scale)
                    img = np.exp(reg_factor * (img - np.ones((img.shape[0], img.shape[1]), dtype=np.float32)))
                else: 
                    # Perform inverse depth
                    img = 1/img
                    img = img/np.amax(img)
                    # Perform norm depth (the prediction is already in log scale)
                    img = np.exp(reg_factor * (img - np.ones((img.shape[0], img.shape[1]), dtype=np.float32)))
                    img = 1/img
                    img = img/np.amax(img)

        if self.color:
            vmax = np.percentile(img, 95)
            normalizer = mpl.colors.Normalize(vmin=img.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            img = mapper.to_rgba(img)
            img = img[:,:,0:3][...,::-1]


        if self.show_events:
            assert(events is not None)
            event_preview = make_event_preview(events, mode=self.event_display_mode,
                                               num_bins_to_show=self.num_bins_to_show)
            event_preview = self.crop_outer_border(event_preview, self.border)

        if self.show_events:
            img_is_color = (len(img.shape) == 3)
            preview_is_color = (len(event_preview.shape) == 3)

            if(preview_is_color and not img_is_color):
                img = np.dstack([img] * 3)
            elif(img_is_color and not preview_is_color):
                event_preview = np.dstack([event_preview] * 3)

            if self.show_reconstruction:
                img = np.hstack([event_preview, img])
            else:
                img = event_preview

        img = np.clip(img, 0.0, 1.0)
        cv2.imshow(self.window_name, img)
        c = cv2.waitKey(self.wait_time)

        if c == ord('s'):
            now = datetime.now()
            path_to_screenshot = '/tmp/screenshot-{}.png'.format(now.strftime("%d-%m-%Y-%H-%M-%S"))
            cv2.imwrite(path_to_screenshot, img)
            print('Saving screenshot to: {}'.format(path_to_screenshot))
        elif c == ord('e'):
            self.show_events = not self.show_events
        elif c == ord('f'):
            self.show_reconstruction = not self.show_reconstruction
