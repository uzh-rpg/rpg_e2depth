import numpy as np
from matplotlib import pyplot as plt
import argparse
from os.path import dirname, join

plt.rcParams.update({'font.size': 22})

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Plotting the framerate profile given a file containing timestamps')

    parser.add_argument('-i', '--timestamps', required=True, type=str)
    parser.add_argument('-w', '--sliding_window_size', default=50, type=int)
    parser.add_argument('-s', '--start_index', default=0, type=int)
    parser.add_argument('-e', '--end_index', default=-1, type=int)

    args = parser.parse_args()

    timestamps = np.loadtxt(args.timestamps)
    timestamps = np.sort(timestamps)

    np.savetxt(join(dirname(args.timestamps), 'timestamps_sorted.txt'), timestamps)

    W = args.sliding_window_size
    start_index = args.start_index
    end_index = args.end_index
    timestamps = timestamps[start_index:end_index]

    mean_framerates = []
    for i in range(W, len(timestamps) - W):
        mean_framerate = float(len(timestamps[i - W: i + W])) / (timestamps[i + W] - timestamps[i - W])
        mean_framerates.append(mean_framerate)

    window_len = 25
    w = np.ones(window_len, np.float64)
    mean_framerates_filtered = np.convolve(w / w.sum(), mean_framerates, mode='same')

    mean_framerate = float(len(timestamps)) / (timestamps[-1] - timestamps[0])
    print('Mean frame rate: {:.1f} Hz'.format(mean_framerate))

    plt.figure(figsize=(8, 8))
    plt.plot(timestamps[W:len(timestamps) - W] - timestamps[0], mean_framerates_filtered / 1000.0)
    # plt.plot(mean_framerates)

    plt.xlabel('Time (s)')
    plt.ylabel('Frame rate (kHz)')
    plt.grid()

    plt.savefig(join(dirname(args.timestamps), 'framerate.pdf'))

    # plt.show()
