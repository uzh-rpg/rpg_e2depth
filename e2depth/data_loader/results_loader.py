from utils.util import first_element_greater_than, last_element_less_than, closest_element_to
from os.path import join
import numpy as np
import cv2
import os
from cvbase.optflow.io import read_flow


MATCHING_TOL = 0.01
# MATCHING_TOL = 0.1


class GroundtruthLoader:
    def __init__(self, folder, start_time_s=0.0, stop_time_s=0.0, load_flow=False, as_gray=True):
        if os.path.exists(join(folder, 'frames')):
            self.folder = join(folder, 'frames')
        else:
            self.folder = folder
        assert(start_time_s >= 0)
        assert(stop_time_s >= 0)
        self.start_time_s = start_time_s
        self.stop_time_s = stop_time_s

        if self.stop_time_s > 0:
            assert(start_time_s <= stop_time_s)

        self.load_flow = load_flow
        if self.load_flow:
            self.flow_folder = join(folder, 'flow')

        self.as_gray = as_gray

        print('Init GroundTruthLoader...')

        # Load the timestamp file
        self.timestamps = np.loadtxt(join(self.folder, 'timestamps.txt'))[:, 1]
        self.initial_timestamp = self.timestamps[0]
        self.relative_timestamps = self.timestamps - self.initial_timestamp

        print('  Initial timestamp: {:.2f} s'.format(self.initial_timestamp))

        # Find the min/max index of the image timestamps
        self.min_idx, self.min_timestamp = first_element_greater_than(
            self.relative_timestamps, self.start_time_s)

        assert(self.min_timestamp is not None), 'last relative stamp ({:.2f} s) < start_time ({:.2f}) s)'.format(
            self.relative_timestamps[-1], self.start_time_s)

        if self.stop_time_s > 0:
            self.max_idx, self.max_timestamp = last_element_less_than(
                self.relative_timestamps, self.stop_time_s)
        else:
            self.max_idx, self.max_timestamp = len(
                self.relative_timestamps) - 1, self.relative_timestamps[-1]

        print('  Min index / stamp: {} / {:.2f} s'.format(self.min_idx, self.min_timestamp))
        print('  Max index / stamp: {} / {:.2f} s'.format(self.max_idx, self.max_timestamp))

        self.relative_timestamps_restricted = self.relative_timestamps[
            self.min_idx:self.max_idx + 1]

        self.current_idx = self.min_idx

    def __len__(self):
        return self.max_idx - self.min_idx + 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx > self.max_idx:
            raise StopIteration
        else:
            img = cv2.imread(
                join(self.folder, 'frame_{:010d}.png'.format(self.current_idx)), cv2.IMREAD_GRAYSCALE if self.as_gray else cv2.IMREAD_UNCHANGED)
            timestamp = self.timestamps[self.current_idx]

            if self.load_flow:
                # the loaded flow is 'backward', i.e. from the current image to to the previous one
                if self.current_idx == 0:
                    flow = None  # no flow map can be computed for the very first image
                else:
                    try:
                        flow = read_flow(join(self.flow_folder, '{:06d}.flo'.format(self.current_idx - 1)))
                    except FileNotFoundError:
                        flow = None

            self.current_idx += 1

            if self.load_flow:
                return (img, timestamp, flow)
            else:
                return (img, timestamp, self.current_idx - 1)


class ReconstructionLoader:
    """
        Generic loader that reads an image from a result folder at a specified time t.
    """

    def __init__(self, folder, as_gray=True, config=None):
        raise NotImplementedError

    def get_reconstruction_common(self, query_timestamp, tol):
        img_idx, timestamp, diff = closest_element_to(
            self.timestamps, query_timestamp)
        if diff > tol:
            raise LookupError(
                '{}: could not find an image close enough to query timestamp: {:.2f} s (closest stamp: {:.2f} s, tol = {:.4f} s, diff = {:.4f} s)'
                .format(type(self).__name__, query_timestamp, timestamp, tol, diff))
        return img_idx, timestamp

    def get_reconstruction(self, query_timestamp, tol):
        """ Get the reconstructed which is closest to timestamp `t` (absolute timestamp, expressed in seconds)
            If the timestamp of the retrieved image is farther than `tol` seconds away from `query_timestamp`, will raise an error.
        """
        raise NotImplementedError


class OursLoader(ReconstructionLoader):
    """
        Load images reconstructed by our learned approach.
    """

    def __init__(self, folder, as_gray=True, config=None):
        # Load timestamps
        self.folder = join(folder)
        self.as_gray = as_gray
        self.timestamps = np.loadtxt(join(self.folder, 'timestamps.txt'))[:, 1]
        assert(len(self.timestamps) > 0)
        print('Init OursLoader...')
        print('   min / max timestamps: {:.2f} s / {:.2f} s'.format(
            self.timestamps[0], self.timestamps[-1]))

    def get_reconstruction(self, query_timestamp, tol=MATCHING_TOL):
        img_idx, timestamp = self.get_reconstruction_common(
            query_timestamp, tol)
        img = cv2.imread(
            join(self.folder, 'frame_{:010d}.png'.format(img_idx)), cv2.IMREAD_GRAYSCALE if self.as_gray else cv2.IMREAD_UNCHANGED)
        return (img, timestamp)


class EventTensorLoader(ReconstructionLoader):
    """
        Load event tensors from a folder containing event tensors.
    """

    def __init__(self, folder, as_gray=True, config=None):
        # Load timestamps
        self.folder = folder
        self.timestamps = np.loadtxt(join(self.folder, 'timestamps.txt'))[:, 1]
        assert(len(self.timestamps) > 0)
        print('Init EventLoader...')
        print('   min / max timestamps: {:.2f} s / {:.2f} s'.format(
            self.timestamps[0], self.timestamps[-1]))

    def get_event_tensor(self, query_timestamp, tol=MATCHING_TOL):
        img_idx, timestamp = self.get_reconstruction_common(
            query_timestamp, tol)
        T = np.load(
            join(self.folder, 'event_tensor_{:010d}.npy'.format(img_idx)))
        return (T, timestamp)


class MRLoader(ReconstructionLoader):
    """
        Load images reconstructed by the Manifold Regularization (MR) approach [1]
        [1]: Munda et al., IJCV'18.
    """

    def __init__(self, folder, as_gray=True, config=None):
        # Load timestamps
        self.folder = folder
        self.as_gray = as_gray
        self.timestamps = np.loadtxt(join(self.folder, 'frametimestamps.txt'))
        # remove last element (which is always corrupted)
        self.timestamps = self.timestamps[:-1]
        assert(len(self.timestamps) > 0)
        print('Init MRLoader...')
        print('   min / max timestamps: {:.2f} s / {:.2f} s'.format(
            self.timestamps[0], self.timestamps[-1]))

    def get_reconstruction(self, query_timestamp, tol=MATCHING_TOL):
        img_idx, timestamp = self.get_reconstruction_common(
            query_timestamp, tol)
        img_idx += 1  # the file naming in MR starts at index 1, and not 0
        img = cv2.imread(
            join(self.folder, 'image{:06d}.png'.format(img_idx)), cv2.IMREAD_GRAYSCALE if self.as_gray else cv2.IMREAD_UNCHANGED)
        return (img, timestamp)


class CFLoader(ReconstructionLoader):
    """
        Load images reconstructed by the complementary filter (CF) approach [1].
        [1]: Scheerlinck et al., ACCV'18
    """

    def __init__(self, folder, as_gray=True, config=None):
        # Load timestamps
        self.folder = folder
        self.timestamps = np.loadtxt(join(self.folder, 'timestamps.txt'))[:, 1]
        self.as_gray = as_gray
        self.config = config

        assert(len(self.timestamps) > 0)
        print('Init CFLoader...')
        print('   min / max timestamps: {:.2f} s / {:.2f} s'.format(
            self.timestamps[0], self.timestamps[-1]))

        self.spatial_filter_sigma = 0.0
        if self.config is not None:
            try:
                self.spatial_filter_sigma = self.config['bilateral_filter_sigma']
            except KeyError:
                self.spatial_filter_sigma = 0.0

        if self.spatial_filter_sigma > 0:
            print('Will apply bilateral filter with sigma={:.3f}'.format(self.spatial_filter_sigma))
        else:
            print('Will not apply bilateral filter')

    def get_reconstruction(self, query_timestamp, tol=MATCHING_TOL):
        img_idx, timestamp = self.get_reconstruction_common(
            query_timestamp, tol)

        img = cv2.imread(
            join(self.folder, 'frame_{:010d}.png'.format(img_idx)), cv2.IMREAD_GRAYSCALE if self.as_gray else cv2.IMREAD_UNCHANGED)

        # (optional) bilateral filter
        if self.spatial_filter_sigma > 0:
            bilateral_filter_sigma = 25 * self.spatial_filter_sigma
            filtered_img = np.zeros_like(img)
            filtered_img = cv2.bilateralFilter(
                img, 5, bilateral_filter_sigma, bilateral_filter_sigma)
        else:
            filtered_img = img

        # (optional) median filter
        # median_filter_size = 3
        # img = cv2.medianBlur(img, median_filter_size)

        return (filtered_img, timestamp)
