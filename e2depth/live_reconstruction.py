import torch
from os.path import join
import numpy as np
import json
import argparse
import time
import socket
import struct
import queue
import threading
from utils.loading_utils import load_model, get_device
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
from utils.timers import Timer, timers

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 8080        # The port used by the server
SIZE_OF_FLOAT = 4


def reconstruction_thread(model, height, width, num_bins, options):
    reconstructor = ImageReconstructor(model, height, width, num_bins, options)
    idx = 0
    while True:
        data = q.get()
        if data is None:
            break

        # Convert the raw byte array encoding the event tensor (voxel grid) into a PyTorch array
        with Timer('Unpack data array'):
            # https://stackoverflow.com/a/36809458
            event_tensor = np.ndarray(shape=(num_bins, height, width), dtype=np.float32, buffer=data, offset=0)
            event_tensor = torch.from_numpy(event_tensor)

        # Unpack timestamp
        stamp = struct.unpack('d', data[-8:])[0]

        reconstructor.update_reconstruction(event_tensor, idx, stamp)
        q.task_done()

        idx += 1


def recvall(sock, n):
    # http://stupidpythonideas.blogspot.com/2013/05/sockets-are-byte-streams-not-message.html
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                        help='path to model checkpoint')

    set_inference_options(parser)

    args = parser.parse_args()

    # Loading model to device
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)

    if args.use_fp16:
        model = model.half()

    model = model.to(device)
    model.eval()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print('Waiting for event tensor server to be ready...')

        connection_established = False
        while not connection_established:
            try:
                s.connect((HOST, PORT))
            except socket.error:
                # print('Connection to server failed. Will try again...')
                time.sleep(0.25)
                continue
            connection_established = True

        s.sendall(b'E2VID client: connected!')

        # Read meta data: a message packing 3 integers encoding the size of the event tensor
        data = recvall(s, 3 * SIZE_OF_FLOAT)
        height, width, num_bins = struct.unpack('3I', data)
        message_length = width * height * num_bins + 2  # last 2 elements encode the timestamp as a double

        print('Received voxel grid size: {} x {} x {}'.format(height, width, num_bins))

        # Launch the reconstruction thread
        q = queue.Queue(maxsize=0)
        t = threading.Thread(target=reconstruction_thread, args=(model, height, width, num_bins, args))
        t.start()

        continue_receiving = True
        while continue_receiving:
            data = recvall(s, message_length * SIZE_OF_FLOAT)
            if data is None:
                continue_receiving = False
                break

            q.put(data)

    # Block until all event tensors have been processed
    q.join()

    # Stop the reconstruction thread
    q.put(None)
    t.join()
