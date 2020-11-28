import glob
import os
import sys
try:
    sys.path.append(glob.glob('/home/javi/carla_sim/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    #sys.path.append(glob.glob('/home/javi/carla_sim/dev/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# Imports
try:
    import carla
    from carla import ColorConverter as cc
except ImportError:
    raise RuntimeError('cannot import carla, make sure carla package is accessible')

try:
    from spawn_npc_thread import run
except ImportError:
    raise RuntimeError('cannot import spawn actors')

import argparse
import random
import time
import weakref
import tqdm
import math
from PIL import Image

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')
try:
    import queue
except ImportError:
    import Queue as queue
try:
    import pandas as pd
except ImportError:
    raise RuntimeError('cannot import pandas, make sure pandas package is installed')
try:
    import cv2
except ImportError:
    raise RuntimeError('cannot import opencv, make sure opencv package is installed')
try:
    import transformations as tf
except ImportError:
    raise RuntimeError('cannot import transformations, make suer the file is in the directory')
# ==============================================================================
def FLAGS():
    parser = argparse.ArgumentParser("CARLA Simulator Depth From Events")
    parser.add_argument(
        '--fov',
        default='83',
        type=float,
        help=' Camera Horizontal Field of View in degrees (default: 83)')
    parser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x480',
        help='window resolution (default: 640x480)')
    parser.add_argument(
        "--output_type",
        default="no_recording",
        help="Type of output for images, can be one of the following: [pandas, numpy, json, folder].")
    parser.add_argument(
        "--output_path",
        default="./",
        help="Path to save the data. (Current directory as default)")
    parser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    parser.add_argument(
        "--preview",
        default=False,
        action='store_true',
        help="Visualize the camera sensors")
    parser.add_argument(
        '--fps',
        default='50.0',
        type=float,
        help='Frame per second of the simulation (50fps by default)')
    parser.add_argument(
        '--dvs_threshold',
        metavar='Positive - Negative',
        default='0.18-0.18',
        help='Dynamic Vision Sensor threshold [Positive]-[Negative] (0.18-0.18 as default)')
    parser.add_argument(
        '--server',
        default='localhost',
        help='Name or address of Carla Server (localhost by default)')
    parser.add_argument(
        '--dvs_sigma_threshold',
        metavar='Positive - Negative',
        default='0.0-0.0',
        help='Dynamic Vision Sensor Sigma threshold [Positive]-[Negative] (1.0-1.0 as default)')
    parser.add_argument(
        '--no_use_log',
        action='store_false',
        help='Use log scale image for creating events')
    parser.add_argument(
        '--log_eps',
        default='0.3',
        type=float,
        help='Epsilon for the log conversion')
    parser.add_argument(
        '--num_bins',
        default='5',
        type=int,
        help='Number of bins in the Voxel Grid (5 bins by default)')
    parser.add_argument(
        "--map",
        default="Town03",
        help="Carla map to load in simulation (Town03 by default)")
    parser.add_argument(
        "--autopilot",
        default=False,
        action='store_true',
        help="Set autopilot")
    parser.add_argument(
        "--number_samples",
        default="-1",
        type=int,
        help="Number of desired samples (infinite loop by default)")
    parser.add_argument(
        "--port",
        default="2000",
        type=int,
        help="Port number to connect to Carla server Listen for client connections at port N, agent ports are set to N+1 and N+2 respectively (default 2000)")
    parser.add_argument(
        "--tm_port",
        default="8000",
        type=int,
        help="Port number to Carla Traffic Manager (default 8000)")
    parser.add_argument(
        "--depth",
        default="cc.LogarithmicDepth",
        help="Depth type of the Depth image: cc.Raw, cc.Depth or cc.LogarithmicDepth (cc.LogarithmicDepth as default)")
    parser.add_argument(
        "--run_id",
        default="-1",
        type = str,
        help="Sequence id for the dataset (timestamp as default)")
    parser.add_argument(
        '-v', '--number_of_vehicles',
        metavar='N',
        default=0,
        type=int,
        help='number of vehicles (default: 0)')
    parser.add_argument(
        '-w', '--number_of_walkers',
        metavar='W',
        default=0,
        type=int,
        help='number of walkers (default: 0)')
    parser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    parser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    parser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    parser.add_argument(
        '--seconds_to_wait',
        default=0.0,
        type=float,
        help='wait x seconds before start recording (default is 0)')
    parser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enanble')

    args = parser.parse_args()

    assert args.output_type in ["pandas", "numpy", "json", "folder", "no_recording"]
    assert os.path.isdir(args.output_path), "%s should be valid dir." % args.output_path
    
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.Cp, args.Cm = [float(x) for x in args.dvs_threshold.split('-')]
    args.sigma_Cp, args.sigma_Cm = [float(x) for x in args.dvs_sigma_threshold.split('-')]

    return args

# ==============================================================================
import threading
from subprocess import call

def get_font(pygame):
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def to_bgra_array(image, height=None, width=None):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    if height is None or width is None:
        array = np.reshape(array, (image.height, image.width, 4))
    else:
        array = np.reshape(array, (height, width, 4))
    return array


def to_rgb_array(image, height=None, width=None):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image, height, width)
    # Convert BGRA (Carla) to RGB (PyGame).
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def depth_to_array(image):
    """
    Convert an numpy array containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = to_bgra_array(image)
    array = array.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth

def events_to_rgb(events, height=None, width=None):
    """Convert an Events to a RGB numpy array."""
    if height is None or width is None:
        array = np.zeros((events.height, events.width, 3), dtype=np.dtype("uint8"))
    else:
        array = np.ones((height, width, 3), dtype=np.dtype("uint8"))
        array *= 255

    if height is None or width is None:
        for i in range(len(events)):
            if (events[i].pol == True):
                array[events[i].y, events[i].x, 2] = 255#Blue
            else:
                array[events[i].y, events[i].x, 0] = 255#Red
    else:
        for i in range(len(events)):
            # [t, x, y, pol] events array
            if (events[i][3] == 1):
                array[int(events[i][2]), int(events[i][1]), :] = [0, 0, 255]#Blue
            else:
                array[int(events[i][2]), int(events[i][1]), :] = [255, 0, 0]#Red
    return array

def events_to_numpy(events):
    """Convert an Events to a [t,x,y,pol] numpy array."""
    array = []
    for e in events:
        if (e.pol == True):
            array.append(np.array([e.t, e.x, e.y, 1], dtype=np.dtype("int64")))
        else:
            array.append(np.array([e.t, e.x, e.y, 0], dtype=np.dtype("int64")))
    return np.asarray(array)

def vector_to_numpy (vector, txyp=True):
    array = []
    for i in range (len(vector)):
        array.append(np.asarray(np.frombuffer(vector[i].raw_data, dtype=np.dtype("int64"))))
    array = np.asarray(array)
    if txyp:
        array = array[:,[2,0,1,3]]
    return array

def events_to_voxel_grid(events, num_bins, height, width):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int)
    ys = events[:, 2].astype(np.int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid

def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def rgb2pygamegray(rgb):
        return rgb.dot([0.298, 0.587, 0.114])[:,:,None].repeat(3,axis=2)

def should_quit(pygame):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def draw_image(surface, image, position=0, blend=False):
    image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (position, 0))

# ==============================================================================
# -- Carla Simulator -----------------------------------------------------------
# ==============================================================================
class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, args):
        self.CARLA_MIN_DELTA_TIME = 0.4 #seconds
        self.DVS_SIMULATION_RATE_FACTOR = 1.0
        self.world = world
        self.sensors = sensors
        self.args = args
        self.frame = None
        self.delta_seconds = 1.0 / (args.fps * self.DVS_SIMULATION_RATE_FACTOR)
        self._queues = []
        self._settings = None

        # Check that FPS is at least > (1/MIN_DELTA)
        if (self.delta_seconds > self.CARLA_MIN_DELTA_TIME):
            self.delta_seconds = self.CARLA_MIN_DELTA_TIME
            self.args.fps = 1.0/self.CARLA_MIN_DELTA_TIME

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        for x in data:
            if x is not None:
                assert x.frame == self.frame
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            try:
                data = sensor_queue.get(timeout=timeout)
                if data.frame == self.frame:
                    return data
            except queue.Empty:
                return None

# ==============================================================================
# -- Recorder ------------------------------------------------------------------
# ==============================================================================
class Recorder(object):

    def __init__(self, args):
        endings = {"numpy": ".npy", "pandas": ".pkl", "json":".json", "folder": ""}
        self.record_ending = endings[args.output_type]
        self.args = args
        # ID and timestamps
        self.id = []
        self.time = []
        # RGB images
        self.image = []
        # Depth data
        self.depth = []
        self.depth_frames = []
        # Events data
        self.events = []
        self.voxels = []
        self.events_frames = []
        # Semantic Segmentation
        self.semantic = []
        self.semantic_frames = []
        # IMU
        self.imu = []
        # Vehicle's pose
        self.pose = []
        if args.run_id is "-1":
            self.path = os.path.join(args.output_path, "sequence_%s%s" % (time.strftime('%Y%m%d_%H%M%S'), self.record_ending))
        else:
            self.path = os.path.join(args.output_path, "sequence_%s%s" % (args.run_id, self.record_ending))
    
    def add_image_frame(self, id, timestamp, data):
        #print ("Recording Image %d at time %f" % (id, timestamp))
        if id in self.id:
            self.image.append(data.copy())
        else:
            self.id.append(id)
            self.time.append(timestamp)
            self.image.append(data.copy())

    def add_depth_data(self, id, timestamp, data):
        #print ("Recording Depth Raw Data %d at time %f" % (id, timestamp))
        if id in self.id:
            self.depth.append(data.copy())
        else:
            self.id.append(id)
            self.time.append(timestamp)
            self.depth.append(data.copy())

    def add_depth_frame(self, id, timestamp, data):
        #print ("Recording Depth Frame %d at time %f" % (id, timestamp))
        if id in self.id:
            self.depth_frames.append(data.copy())
        else:
            self.id.append(id)
            self.time.append(timestamp)
            self.depth_frames.append(data.copy())

    def add_events_data(self, id, timestamp, data):
        #print ("Recording DVS Data %d at time %f" % (id, timestamp))
        if id in self.id:
            # already exist the id
            self.events.append(data.copy())
        else:
            self.id.append(id)
            self.time.append(timestamp)
            self.events.append(data.copy())

    def add_events_frame(self, id, timestamp, data):
        #print ("Recording DVS Frame %d at time %f" % (id, timestamp))
        if id in self.id:
            # already exist the id
            self.events_frames.append(data.copy())
        else:
            self.id.append(id)
            self.time.append(timestamp)
            self.events_frames.append(data.copy())

    def add_events_voxel(self, id, timestamp, data):
        #print ("Recording VoxelGrid %d at time %f" % (id, timestamp))
        voxel = events_to_voxel_grid(data, self.args.num_bins, self.args.height, self.args.width)
        if id in self.id:
            # already exist the id
            self.voxels.append(voxel.copy())
        else:
            self.id.append(id)
            self.time.append(timestamp)
            self.voxels.append(voxel.copy())

    def add_semantic_data(self, id, timestamp, data):
        #print ("Recording Semantic Raw Data %d at time %f" % (id, timestamp))
        if id in self.id:
            self.semantic.append(data.copy())
        else:
            self.id.append(id)
            self.time.append(timestamp)
            self.semantic.append(data.copy())

    def add_semantic_frame(self, id, timestamp, data):
        #print ("Recording Semantic Frame %d at time %f" % (id, timestamp))
        if id in self.id:
            self.semantic_frames.append(data.copy())
        else:
            self.id.append(id)
            self.time.append(timestamp)
            self.semantic_frames.append(data.copy())

    def add_imu_data(self, id, timestamp, data):
        #print ("Recording IMU Data %d at time %f" % (id, timestamp))
        if id in self.id:
            self.imu.append(data.copy())
        else:
            self.id.append(id)
            self.time.append(timestamp)
            self.imu.append(data.copy())

    def add_pose_data(self, id, timestamp, data):
        #print ("Recording Vehicle's Pose Data %d at time %f" % (id, timestamp))
        if id in self.id:
            self.pose.append(data.copy())
        else:
            self.id.append(id)
            self.time.append(timestamp)
            self.pose.append(data.copy())

    def save(self):
        self.min_len = min([len(self.id), len(self.time), len(self.image), len(self.depth), len(self.events)])
        print("Saving %d samples in this experiment run" % self.min_len)

        df = pd.DataFrame({'id':self.id[:self.min_len],
                            'time': self.time[:self.min_len] ,
                            'imu': self.imu[:self.min_len] ,
                            'pose': self.pose[:self.min_len] ,
                            'image':self.image[:self.min_len],
                            'depth':self.depth[:self.min_len],
                            'depth_frames':self.depth_frames[:self.min_len],
                            'events':self.events[:self.min_len],
                            'events_frames':self.events_frames[:self.min_len],
                            'voxels':self.voxels[:self.min_len],
                            'semantic':self.semantic[:self.min_len],
                            'semantic_frames':self.semantic_frames[:self.min_len]}
                            )
        if self.path.endswith(".pkl"):
            df.to_pickle(self.path)
        elif self.path.endswith(".npy"):
            np.save(self.path, df.values)
        elif self.path.endswith(".json"):
            df.to_json(self.path)
        elif self.path.endswith(""):
            self.save_to_folder(df)
        else:
            raise ValueError("Path must end with either .pkl or .npy")

    def save_to_folder(self, df):
        # Folder for RGB images
        img_folder = os.path.join(self.path, "rgb")
        os.system("mkdir -p %s" % img_folder)
        img_folder_frames = os.path.join(img_folder, "frames")
        os.system("mkdir -p %s" % img_folder_frames)

        # Folder for depth informatiom
        depth_folder = os.path.join(self.path, "depth")
        os.system("mkdir -p %s" % depth_folder)
        depth_folder_data = os.path.join(depth_folder, "data")
        os.system("mkdir -p %s" % depth_folder_data)
        depth_folder_frames = os.path.join(depth_folder, "frames")
        os.system("mkdir -p %s" % depth_folder_frames)

        # Folder for events information
        events_folder = os.path.join(self.path, "events")
        os.system("mkdir -p %s" % events_folder)
        events_folder_data = os.path.join(events_folder, "data")
        os.system("mkdir -p %s" % events_folder_data)
        events_folder_frames = os.path.join(events_folder, "frames")
        os.system("mkdir -p %s" % events_folder_frames)
        events_folder_voxels = os.path.join(events_folder, "voxels")
        os.system("mkdir -p %s" % events_folder_voxels)

        # Folder for semantic informatiom
        semantic_folder = os.path.join(self.path, "semantic")
        os.system("mkdir -p %s" % semantic_folder)
        semantic_folder_data = os.path.join(semantic_folder, "data")
        os.system("mkdir -p %s" % semantic_folder_data)
        semantic_folder_frames = os.path.join(semantic_folder, "frames")
        os.system("mkdir -p %s" % semantic_folder_frames)

        # Folder for the pose
        pose_folder = os.path.join(self.path, "pose")
        os.system("mkdir -p %s" % pose_folder)
        pose_folder_imu = os.path.join(pose_folder, "imu")
        os.system("mkdir -p %s" % pose_folder_imu)
        pose_folder_vehicle = os.path.join(pose_folder, "vehicle")
        os.system("mkdir -p %s" % pose_folder_vehicle)

        # Save pose (imu and vehicle's pose) data first
        imu_name = "imu_measurements.npy"
        np.save(os.path.join(pose_folder_imu, imu_name), df['imu'].to_numpy())
        pose_name = "vehicle_pose_measurements.npy"
        np.save(os.path.join(pose_folder_vehicle, pose_name), df['pose'].to_numpy())

        pbar = tqdm.tqdm(total=self.min_len)
        print ("\nSaving dataset to folder")
        for _, row in df.iterrows():
            frame_name = "frame_%010d.png" % int(row['id'])
            #pbar.write(frame_name)

            # Save RGB images
            img = row["image"]
            cv2.imwrite(os.path.join(img_folder_frames, frame_name),cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) # OpenCV reads images in BGR or BGRA and pygame and PIL in RGB

            #Save Depth data
            depth = row["depth"]
            depth_name = "depth_%010d.npy" % int(row['id'])
            np.save(os.path.join(depth_folder_data, depth_name), depth)

            #Save Depth frames
            depth_frame = row["depth_frames"]
            if eval(self.args.depth) is cc.Raw:
                cv2.imwrite(os.path.join(depth_folder_frames, frame_name),cv2.cvtColor(depth_frame, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(os.path.join(depth_folder_frames, frame_name),cv2.cvtColor(depth_frame, cv2.COLOR_RGB2GRAY))

            # Save Events data
            events = row["events"]
            events_name = "events_%010d.npy" % int(row['id'])
            np.save(os.path.join(events_folder_data, events_name), events)

            # Events Frames folder
            events_img = row['events_frames']
            cv2.imwrite(os.path.join(events_folder_frames, frame_name),cv2.cvtColor(events_img, cv2.COLOR_RGB2BGR)) # OpenCV read images in BGR

            # Voxel Grid
            voxel = row["voxels"]
            voxel_name = "event_tensor_%010d.npy" % int(row['id'])
            np.save(os.path.join(events_folder_voxels, voxel_name), voxel)

            #Save Semantic data
            semantic = row["semantic"]
            semantic_name = "semantic_%010d.npy" % int(row['id'])
            np.save(os.path.join(semantic_folder_data, semantic_name), semantic)

            #Save Semantic frames
            semantic_frame = row["semantic_frames"]
            cv2.imwrite(os.path.join(semantic_folder_frames, frame_name),cv2.cvtColor(semantic_frame, cv2.COLOR_RGB2BGR))

            # Update the moving bar
            pbar.update(1)

            # Save all the timestamps in separated files
            with open(os.path.join(img_folder_frames, "timestamps.txt"), "a") as f_handle:
                f_handle.write("%s %s\n" % (int(row['id']), float(row["time"]), ))

            # Save all the timestamps in separated files
            with open(os.path.join(depth_folder_data, "timestamps.txt"), "a") as f_handle:
                f_handle.write("%s %s\n" % (int(row['id']), float(row["time"]), ))
            with open(os.path.join(depth_folder_frames, "timestamps.txt"), "a") as f_handle:
                f_handle.write("%s %s\n" % (int(row['id']), float(row["time"]), ))

            # Save all the timestamps in separated files
            with open(os.path.join(events_folder_data, "timestamps.txt"), "a") as f_handle:
                f_handle.write("%s %s\n" % (int(row['id']), float(row["time"]), ))
            with open(os.path.join(events_folder_data, "boundary_timestamps.txt"), "a") as f_handle:
                f_handle.write("%s %s %s\n" % (int(row['id']), float( np.min(row['events'][:,0], axis=0)/1e9),  float( np.max(row['events'][:,0], axis=0)/1e9)))
            with open(os.path.join(events_folder_frames, "timestamps.txt"), "a") as f_handle:
                f_handle.write("%s %s\n" % (int(row['id']), float(row["time"]), ))
            with open(os.path.join(events_folder_frames, "boundary_timestamps.txt"), "a") as f_handle:
                f_handle.write("%s %s %s\n" % (int(row['id']), float( np.min(row['events'][:,0], axis=0)/1e9),  float( np.max(row['events'][:,0], axis=0)/1e9)))
            with open(os.path.join(events_folder_voxels, "timestamps.txt"), "a") as f_handle:
                f_handle.write("%s %s\n" % (int(row['id']), float(row["time"]), ))
            with open(os.path.join(events_folder_voxels, "boundary_timestamps.txt"), "a") as f_handle:
                f_handle.write("%s %s %s\n" % (int(row['id']), float( np.min(row['events'][:,0], axis=0)/1e9),  float( np.max(row['events'][:,0], axis=0)/1e9)))

            # Save all the timestamps in separated files
            with open(os.path.join(semantic_folder_data, "timestamps.txt"), "a") as f_handle:
                f_handle.write("%s %s\n" % (int(row['id']), float(row["time"]), ))
            with open(os.path.join(semantic_folder_frames, "timestamps.txt"), "a") as f_handle:
                f_handle.write("%s %s\n" % (int(row['id']), float(row["time"]), ))
            
            # Save all the timestamps in separated files
            with open(os.path.join(pose_folder_imu, "timestamps.txt"), "a") as f_handle:
                f_handle.write("%s %s\n" % (int(row['id']), float(row["time"]), ))
            with open(os.path.join(pose_folder_vehicle, "timestamps.txt"), "a") as f_handle:
                f_handle.write("%s %s\n" % (int(row['id']), float(row["time"]), ))


# ==============================================================================
# -- Main loop -----------------------------------------------------------------
# ==============================================================================
def main(args, number_surfaces = 3):
    actor_list = []
    accumulate_id = 0
    record = False
    frame_id = 0
    pygame.init()
    save_delta_seconds = 1.0/ args.fps
    accumulative_delta_seconds = 0
    array_events = np.array([], ndmin=2)
    number_samples = 0
    color_converter = eval(args.depth)
    if args.number_samples is -1:
        number_samples = -2

    if args.preview:
        display = pygame.display.set_mode(
            ((number_surfaces) * args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('RPG - Event2Depth - Carla Dataset')

    font = get_font(pygame)
    clock = pygame.time.Clock()

    client = carla.Client(args.server, args.port)
    client.set_timeout(10.0)
    client.load_world(args.map)

    world = client.get_world()
    weather = world.get_weather()
    #world.set_weather(carla.WeatherParameters.ClearSunset)
    weather.sun_altitude_angle = random.randint(0, 180)
    world.set_weather(weather)

    if args.output_type is not "no_recording":
        recorder = Recorder(args)
        record = True

    no_wait = (False if args.seconds_to_wait != 0.0 else True)

    # Print info about weather
    print(world.get_weather())

    try:
        # Get the map and the starting point of the car
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())

        # The libraries        
        blueprint_library = world.get_blueprint_library()

        # Create our fancy car
        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.tesla.model3')),
            start_pose)
        vehicle.set_simulate_physics(True)
        # Append the car to the list of actors
        actor_list.append(vehicle)

        # Set Autopilot
        if args.autopilot:
            vehicle.apply_control(carla.VehicleControl(throttle=0.1, brake=0.1))
            vehicle.set_autopilot(True, args.tm_port)
        else:
            waypoint = m.get_waypoint(start_pose.location)

        # Create a RGB camera
        sensor = blueprint_library.find('sensor.camera.rgb')
        sensor.set_attribute("image_size_x", str(args.width))
        sensor.set_attribute("image_size_y", str(args.height))
        sensor.set_attribute("fov", str(args.fov))
        sensor.set_attribute('exposure_mode', str('manual'))
        print("rgb exposure_mode", sensor.get_attribute('exposure_mode'))
        sensor.set_attribute('exposure_min_bright', str(0.1))
        print("rgb exposure_min_bright", sensor.get_attribute('exposure_min_bright'))
        sensor.set_attribute('exposure_max_bright', str(2.0))
        print("rgb exposure_max_bright", sensor.get_attribute('exposure_max_bright'))
        print("enable_postprocess_effects", sensor.get_attribute('enable_postprocess_effects'))
        #Lens parameters
        sensor.set_attribute('lens_k', str(0.0))
        sensor.set_attribute('lens_x_size', str(0.0))
        sensor.set_attribute('lens_y_size', str(0.0))

        camera_rgb = world.spawn_actor( sensor,
            carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        # Create a DVS camera
        sensor = blueprint_library.find('sensor.camera.dvs')
        sensor.set_attribute("image_size_x", str(args.width))
        sensor.set_attribute("image_size_y", str(args.height))
        sensor.set_attribute("fov", str(args.fov))
        sensor.set_attribute('positive_threshold', str(args.Cp))
        sensor.set_attribute('negative_threshold', str(args.Cm))
        sensor.set_attribute('sigma_positive_threshold', str(args.sigma_Cp))
        sensor.set_attribute('sigma_negative_threshold', str(args.sigma_Cm))
        sensor.set_attribute('use_log', str(args.no_use_log))
        sensor.set_attribute('log_eps', str(args.log_eps))
        print("dvs threshold p:%s n:%s" % (sensor.get_attribute('positive_threshold').as_float(), sensor.get_attribute('negative_threshold').as_float()))
        print("dvs sigma threshold p:%s n:%s" % (sensor.get_attribute('sigma_positive_threshold').as_float(), sensor.get_attribute('sigma_negative_threshold').as_float()))
        print("dvs use log: %s (epsilon: %s) " % (sensor.get_attribute('use_log').as_bool(), sensor.get_attribute('log_eps').as_float()))
        sensor.set_attribute('motion_blur_intensity', str(0.0))
        print("dvs motion_blur_intensity", sensor.get_attribute('motion_blur_intensity'))
        sensor.set_attribute('motion_blur_max_distortion', str(0.0))
        print("dvs motion_blur_max_distortion", sensor.get_attribute('motion_blur_max_distortion'))
        #Lens parameters
        sensor.set_attribute('lens_k', str(0.0))
        sensor.set_attribute('lens_x_size', str(0.0))
        sensor.set_attribute('lens_y_size', str(0.0))

        camera_dvs = world.spawn_actor(sensor,
            carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0)),
            attach_to=vehicle)
        actor_list.append(camera_dvs)

        # Create a Depth camera
        sensor = blueprint_library.find('sensor.camera.depth')
        sensor.set_attribute("image_size_x", str(args.width))
        sensor.set_attribute("image_size_y", str(args.height))
        sensor.set_attribute("fov", str(args.fov))
        #Lens parameters
        sensor.set_attribute('lens_k', str(0.0))
        sensor.set_attribute('lens_x_size', str(0.0))
        sensor.set_attribute('lens_y_size', str(0.0))

        camera_depth = world.spawn_actor(sensor,
            carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0)),
            attach_to=vehicle)
        actor_list.append(camera_depth)

        # Create a Semantic Segmentation camera
        sensor = blueprint_library.find('sensor.camera.semantic_segmentation')
        sensor.set_attribute("image_size_x", str(args.width))
        sensor.set_attribute("image_size_y", str(args.height))
        sensor.set_attribute("fov", str(args.fov))
        #Lens parameters
        sensor.set_attribute('lens_k', str(0.0))
        sensor.set_attribute('lens_x_size', str(0.0))
        sensor.set_attribute('lens_y_size', str(0.0))

        camera_semantic = world.spawn_actor(sensor,
            carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0)),
            attach_to=vehicle)
        actor_list.append(camera_semantic)

        # Create IMU sensor
        sensor = blueprint_library.find('sensor.other.imu')
        imu_sensor = world.spawn_actor(sensor,
            carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0)),
            attach_to=vehicle)
        actor_list.append(imu_sensor)

        cars_and_walkers = threading.Thread(target=run, args=(args, client))
        if args.number_of_vehicles > 0 and args.number_of_walkers > 0:
            cars_and_walkers.start()

        #for blueprint in blueprint_library:
        #    print(blueprint)
        #    for attribute in blueprint:
        #        print('  - %s' % attribute)

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_dvs, camera_depth, camera_semantic, imu_sensor, args=args) as sync_mode:
            pbar = tqdm.tqdm(total=args.number_samples)
            while number_samples < args.number_samples:
                if should_quit(pygame):
                    return

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, dvs_events, image_depth, image_semantic, imu_readings = sync_mode.tick(timeout=2.0)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # vehicle next point 
                if not args.autopilot:
                    point = random.uniform(0.001, 0.025)
                    #print (point)
                    waypoint = random.choice(waypoint.next(point))
                    vehicle.set_transform(waypoint.transform)

                # Get the RGB Image
                array_rgb = to_rgb_array(image_rgb)

                # Get the DVS frame
                #print("events ", dvs_events)
                if dvs_events is not None:
                    array_dvs = np.frombuffer(dvs_events.raw_data, dtype=np.dtype([
                        ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
                    #array_dvs = to_rgb_array(dvs_frame, dvs_events.height, dvs_events.width)

                # This has to be done here (before the other color converter is done unless color_converter is cc.Raw)
                if record:
                    # Get the Raw Depth Image
                    image_depth.convert(cc.Raw)
                    array_raw_depth = depth_to_array(image_depth) # Normalized depth (0-1)

                # Get the Depth image frame in the selected color converter
                image_depth.convert(color_converter)
                array_depth = to_rgb_array(image_depth)

                # This has to be done here (before the other color converter is done unless color_converter is cc.Raw)
                if record:
                    # Get the Raw Semantic Image
                    image_semantic.convert(cc.Raw)
                    array_raw_semantic = to_rgb_array(image_semantic)[:,:,0] # Semantic in in red channel

                # Get the Semantic image frame in the selected color converter
                image_semantic.convert(cc.CityScapesPalette)
                array_semantic = to_rgb_array(image_semantic)

                # Sum the delta time from simulation
                accumulative_delta_seconds += snapshot.timestamp.delta_seconds

                if args.preview:
                    # Draw RGB image
                    draw_image(display, array_rgb, position=0*args.width)

                    # Draw Depth image
                    draw_image(display, array_depth, position=1*args.width, blend=False)
 
                    # Draw event frame
                    if dvs_events is not None:
                        dvs_img = np.zeros((dvs_events.height, dvs_events.width, 3), dtype=np.uint8)
                        # Blue is positive, red is negative
                        dvs_img[array_dvs[:]['y'], array_dvs[:]['x'], array_dvs[:]['pol'] * 2] = 255
                        draw_image(display, dvs_img, position=0*args.width,
                                blend=True)
 
                    # Draw Semantic image
                    draw_image(display, array_semantic, position=2*args.width, blend=False)

                    if accumulative_delta_seconds >= save_delta_seconds:
                        clock.tick()
                        #display.blit(
                        #    font.render('% 5f FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                        #    (8, 10))
                        #display.blit(
                        #    font.render('% 5f FPS (simulated)' % round(fps/sync_mode.DVS_SIMULATION_RATE_FACTOR), True, (255, 255, 255)),
                        #    (8, 28))
                        pygame.display.flip()


                if record and no_wait:

                    if accumulative_delta_seconds >= save_delta_seconds:
                        # Record IMU readings
                        imu_data = np.array([imu_readings.timestamp, imu_readings.gyroscope.x, imu_readings.gyroscope.y, imu_readings.gyroscope.z,
                                    imu_readings.accelerometer.x, imu_readings.accelerometer.y, imu_readings.accelerometer.z,
                                    imu_readings.compass], dtype=np.float32)
                        recorder.add_imu_data(frame_id, imu_readings.timestamp, imu_data)

                        # Record vehicle pose
                        lin_velocity = vehicle.get_velocity()
                        ang_velocity = vehicle.get_angular_velocity()
                        transform = vehicle.get_transform().get_matrix()
                        q = tf.quaternion_from_matrix(np.array(transform)) # [x y z w]
                        translation = np.array([transform[0][3], transform[1][3], transform[2][3]])
                        pose_data = np.array([imu_readings.timestamp, np.array([ang_velocity.x, ang_velocity.y, ang_velocity.z]),
                                              np.array([lin_velocity.x, lin_velocity.y, lin_velocity.z]),
                                              q, translation])
                        recorder.add_pose_data(frame_id, imu_readings.timestamp, pose_data)

                    # Record RGB images
                    if accumulative_delta_seconds >= save_delta_seconds:
                        recorder.add_image_frame(frame_id, image_rgb.timestamp, array_rgb)

                    # Accumulate DVS events in array
                    if dvs_events is not None:
                        x = np.asarray(dvs_events.to_array_x())
                        y = np.asarray(dvs_events.to_array_y())
                        t = np.asarray(dvs_events.to_array_t())
                        pol = np.asarray(dvs_events.to_array_pol())
                        array = np.asarray([t, x, y, pol]) # order expected for array_to_voxel_grid
                        array = array.transpose()
                        array_events = np.append(array_events, array)
                        array_events = np.reshape(array_events, (int(len(array_events)/4), 4))
                        accumulate_id += 1
                        # Record DVS array, events frame and voxel grid
                        if accumulative_delta_seconds >= save_delta_seconds:
                            # Save the events data
                            recorder.add_events_data(frame_id, dvs_events.timestamp, array_events)
                            # Save the events frame
                            img_events = events_to_rgb(array_events, dvs_events.height, dvs_events.width)
                            recorder.add_events_frame(frame_id, dvs_events.timestamp, img_events)
                            #recorder.add_events_frame(frame_id, dvs_events.timestamp, array_dvs)
                            # Save the events voxel
                            recorder.add_events_voxel(frame_id, dvs_events.timestamp, array_events)
                            # reset the accumulative events
                            array_events = np.array([])
                            accumulate_id = 0

                    # Record Depth
                    if accumulative_delta_seconds >= save_delta_seconds:
                        # Record Depth Raw data (meters)
                        carla_far_plane = 1000.0
                        recorder.add_depth_data(frame_id, image_depth.timestamp, array_raw_depth * carla_far_plane)

                        # Record Depth frames in the selected color converter in arguments
                        recorder.add_depth_frame(frame_id, image_depth.timestamp, array_depth)

                    # Record Semantic
                    if accumulative_delta_seconds >= save_delta_seconds:
                        # Record Semantic Raw data (red channel)
                        recorder.add_semantic_data(frame_id, image_semantic.timestamp, array_raw_semantic)
 
                        # Record Semantic frames in CitySpacePalette
                        recorder.add_semantic_frame(frame_id, image_semantic.timestamp, array_semantic)

                        # Increase frame id
                        frame_id += 1

                #print("image timestamp:", image_rgb.timestamp)
                #print("dvs timestamp:", dvs_events.timestamp)
                #print("----------------")
                if no_wait is False:
                    if accumulative_delta_seconds >= args.seconds_to_wait:
                        no_wait = True
                elif accumulative_delta_seconds >= save_delta_seconds:
                    # reset delta seconds time
                    accumulative_delta_seconds = 0
                    if args.number_samples is not -1:
                        number_samples += 1
                        pbar.update(1)

                # Move the Sun location (optional)
                #weather = world.get_weather()
                #weather.sun_altitude_angle += (random.randint(0, 1) * 0.05)
                #world.set_weather(weather)

    finally:
        # Stop the sync mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        if record:
            recorder.save()
 
        print('destroying actors.')
        for actor in actor_list:
            if actor.is_alive:
                print ("\tactor %s"%actor)
                actor.set_simulate_physics(False)
                actor.destroy()
        print('done.')

        if args.number_of_vehicles > 0 and args.number_of_walkers > 0:
            cars_and_walkers.do_run = False
            cars_and_walkers.join()
      
        pygame.quit()
# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================
if __name__ == '__main__':
    flags = FLAGS()

    try:

        main(flags)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
