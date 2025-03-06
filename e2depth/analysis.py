import numpy as np
from os.path import join
import cv2
from matplotlib import pyplot as plt
from data_loader.results_loader import GroundtruthLoader
from data_loader.results_loader import *
from model.metric import mse, structural_similarity, perceptual_distance
import argparse
import json
from utils.path_utils import ensure_dir
from skimage.exposure import equalize_adapthist, equalize_hist
import os
import pickle


def convert_to_8bit(img):
    img_8bit = 255.0 * img.copy()
    return img_8bit.astype(np.uint8)


def convert_image(img):
    assert(img.dtype == np.uint8)
    assert(len(img.shape) == 2)  # image should be [H x W]
    img = img.astype(np.float32)
    img /= 255  # [H x W]
    # convert from [H x W] to [1 x 1 x H x W]
    return img.reshape((1, 1, img.shape[0], img.shape[1]))


def crop_image(img, crop_size):
    H, W = img.shape[0], img.shape[1]
    return img[crop_size:H - crop_size, crop_size: W - crop_size]


def remove_NaNs(L):
    return [elt for elt in L if not np.isnan(elt)]


def equalize(img, global_equalization=False):
    assert(img.dtype == np.uint8)

    if global_equalization:
        # Global histogram equalization
        return cv2.equalizeHist(img)
    else:
        # Local histogram equalization
        img = img.astype(np.float64) / 255.0
        img = equalize_adapthist(img)
        img *= 255.0
        return img.astype(np.uint8)


def temporal_consistency_loss(image0, image1, processed0, processed1, flow01, alpha=300.0, output_images=False):
    # assert(image0.dtype == np.float)
    # assert(image1.dtype == np.float)
    # assert(processed0.dtype == np.float)
    # assert(processed1.dtype == np.float)
    assert(flow01.shape[2] == 2)

    # flow01: [H x W x 2] -> [2 x W x H] -> [2 x H x W]
    flow01 = np.swapaxes(flow01, 0, 2)
    flow01 = np.swapaxes(flow01, 1, 2)

    height, width = image0.shape[:2]
    xx, yy = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))
    x = np.vstack([xx.flatten(), yy.flatten(), np.ones(width * height)])
    pts_coords1 = x[:2, :].reshape((2, height, width))  # coordinates of all points on the image plane
    pts_coords1_warped_to0 = pts_coords1 + flow01  # coords of each pixel in image1 warped to image0
    map01_x = pts_coords1_warped_to0[0, :, :].reshape((height, width)).astype(np.float32)
    map01_y = pts_coords1_warped_to0[1, :, :].reshape((height, width)).astype(np.float32)

    image1_warped_to0 = np.zeros_like(image0)
    processed1_warped_to0 = np.zeros_like(processed0)
    cv2.remap(src=image1, dst=image1_warped_to0, map1=map01_x, map2=map01_y, interpolation=cv2.INTER_LINEAR)
    cv2.remap(src=processed1, dst=processed1_warped_to0, map1=map01_x, map2=map01_y, interpolation=cv2.INTER_LINEAR)

    visibility_mask = np.exp(-alpha * (image0 - image1_warped_to0) ** 2)
    kernel = np.ones((3, 3), np.uint8)
    visibility_mask = cv2.erode(visibility_mask, kernel, iterations=1)

    error_map = visibility_mask * np.fabs(processed1_warped_to0 - processed0)
    total_error = np.mean(error_map)

    if output_images:
        additional_output = {'image1_warped_to0': image1_warped_to0,
                             'processed1_warped_to0': processed1_warped_to0,
                             'visibility_mask': visibility_mask,
                             'error_map': error_map}
        return total_error, additional_output
    else:
        return total_error


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Compare our approach with others, quantitatively')
    parser.add_argument(
        '--config', default='analysis/ours_only.json', help="Path to config file")
    parser.add_argument('--gt_folder', required=True, type=str,
                        help="Path to the groundtruth folder")
    parser.add_argument('--output_folder', required=True,
                        type=str, help="Path to the output folder")
    parser.add_argument('--border_crop_size', default=0,
                        type=int, help="Crop the outer border by N pixels")
    parser.add_argument('--num_plots', default=5, type=int,
                        help="Number of preview images to plot / save to the output folder")
    parser.add_argument('--eval_temporal_error', dest='eval_temporal_error',
                        action='store_true')
    parser.set_defaults(eval_temporal_error=False)
    parser.add_argument('--no-equalize_histogram', dest='no_equalize_histogram',
                        action='store_true')
    parser.set_defaults(no_equalize_histogram=False)

    args = parser.parse_args()

    print_every_n = 50

    if args.config is not None:
        # we will evaluate multiple datasets
        # with the respective configs defined in the config file
        config = json.load(open(args.config))

        groundtruth_folder = args.gt_folder
        base_output_folder = args.output_folder
        base_reconstruction_folder = os.environ['IMAGE_RECONSTRUCTIONS_FOLDER']

        datasets = config['datasets']

        ensure_dir(base_output_folder)
        metrics_file = open(join(base_output_folder, 'metrics.txt'), 'w')
        if args.eval_temporal_error:
            temporal_errors_file = open(join(base_output_folder, 'temporal_errors.txt'), 'w')

        method_names = [method_config['shortname'] for method_config in config['methods']]
        print(method_names)

        metric_funcs = [mse, structural_similarity, perceptual_distance]
        metric_names = ['MSE', 'SSIM', 'LPIPS']

        # Stores the mean metric values over all the datasets, for each metric and each method
        means = {metric_name: {method_name: []
                               for method_name in method_names} for metric_name in metric_names}

        # Stores the mean temporal errors over all the datasets
        if args.eval_temporal_error:
            means_temporal_errors = {method_name: [] for method_name in ['GT', *method_names]}

        """This dictionary will contain the final results, i.e. the mean metric value for every metric, method and dataset
           It will have the following form:
                saved_results =
                    {
                        "datasetA": {
                            "metric1": {
                                "methodA": val, "methodB": val, "methodC": val
                                },
                            "metric2": {
                                "methodA": val, "methodB": val, "methodC": val
                                },
                            "temporal_error": {
                                "groundtruth": val, "methodA": val, "methodC": val
                                }
                        },
                        "datasetB": {
                            "metric1": {
                                "methodA": val, "methodB": val, "methodC": val
                                },
                            "metric2": {
                                "methodA": val, "methodB": val, "methodC": val
                                },
                            "temporal_error": {
                                "groundtruth": val, "methodA": val, "methodC": val
                                }
                        }
                    }
        """
        saved_results = {}

        for dataset in datasets:
            dataset_name = dataset['name']
            start_time_s = dataset['config']['start_time_s']
            stop_time_s = dataset['config']['stop_time_s']

            output_folder = join(base_output_folder, dataset_name)
            ensure_dir(output_folder)
            ensure_dir(join(output_folder, 'raw'))
            ensure_dir(join(output_folder, 'processed'))

            if args.eval_temporal_error:
                ensure_dir(join(output_folder, 'temporal_error'))

            print('Processing dataset: {} (start/stop time: {:.1f}/{:.1f} s'.format(
                dataset_name, start_time_s, stop_time_s))

            # Instantiate ground truth image loader
            groundtruth_loader = GroundtruthLoader(
                join(groundtruth_folder, dataset_name),
                start_time_s, stop_time_s, as_gray=True, load_flow=args.eval_temporal_error)

            # Instantiate image loaders for each method
            method_loaders = []
            for method_config in config['methods']:
                path_to_method_folder = join(base_reconstruction_folder, method_config['folder'], dataset_name)
                loader = eval(method_config['loader_type'])(path_to_method_folder, as_gray=True,
                                                            config=method_config['loader_config'] if 'loader_config' in method_config else None)
                method_loaders.append(loader)

            """Initialize a dictionary to store the result, with the following structure:
                results =
                    {
                        "metric1": { "methodA": [], "methodB": [], "methodC": [] },
                        "metric2": { "methodA": [], "methodB": [], "methodC": [] }
                    }
            """
            results = {metric_name: {method_name: []
                                     for method_name in method_names} for metric_name in metric_names}

            """Initialize a dictionary to store the temporal errors, with the following structure:
                temporal_errors =
                    {
                        "methodA": [],
                        "methodB": [],
                        "methodC": [],
                        "GT": []
                    }
            """
            if args.eval_temporal_error:
                temporal_errors = {method_name: [] for method_name in method_names}
                temporal_errors['GT'] = []

            # Dictionary containing the last image for each method (and the ground truth).
            # Used to compute the temporal consistency loss between pairs of frames.
            if args.eval_temporal_error:
                last_images = {method_name: None for method_name in method_names}
                last_images['GT'] = None

            previews = {method_name: []
                        for method_name in [*method_names, 'groundtruth']}
            previews['groundtruth'] = []

            timestamps_file = open(join(output_folder, 'timestamps.txt'), 'w')

            plot_every_n = int(len(groundtruth_loader) / args.num_plots)

            # Iterate through the ground truth images and evaluate all metrics, for all methods tested
            for image_idx, item in enumerate(groundtruth_loader):

                if args.eval_temporal_error:
                    groundtruth_img, groundtruth_timestamp, flow = item
                else:
                    groundtruth_img, groundtruth_timestamp, _ = item

                groundtruth_img = crop_image(groundtruth_img, args.border_crop_size)

                if image_idx % print_every_n == 0:
                    print('Image: {} / {}'.format(image_idx,
                                                  len(groundtruth_loader) - 1))

                assert(groundtruth_img.dtype == np.uint8)

                if image_idx % plot_every_n == 0:
                    cv2.imwrite(join(output_folder, 'raw', '{}_groundtruth.png').format(
                        image_idx), groundtruth_img)

                if not args.no_equalize_histogram:
                    groundtruth_img = equalize(groundtruth_img)

                if image_idx % plot_every_n == 0:
                    cv2.imwrite(join(output_folder, 'processed', '{}_groundtruth.png').format(
                        image_idx), groundtruth_img)

                    print('{} {:.10f}'.format(
                        image_idx, groundtruth_timestamp), file=timestamps_file)

                groundtruth_img = convert_image(groundtruth_img)

                # only used by temporal_consistency_loss which expects a [H x W] image
                if args.eval_temporal_error:
                    groundtruth_img1 = groundtruth_img[0, 0, :, :]

                # temporal error for ground truth
                if args.eval_temporal_error:
                    groundtruth_img0 = last_images['GT']
                    if groundtruth_img0 is not None:
                        temporal_errors['GT'].append(temporal_consistency_loss(groundtruth_img0, groundtruth_img1,
                                                                               groundtruth_img0, groundtruth_img1,
                                                                               flow))
                    last_images['GT'] = groundtruth_img1.copy()

                if image_idx % plot_every_n == 0:
                    previews['groundtruth'].append(
                        groundtruth_img[0, 0].copy())

                for method_name, loader in zip(method_names, method_loaders):

                    try:
                        img, timestamp = loader.get_reconstruction(
                            groundtruth_timestamp)
                        img = crop_image(img, args.border_crop_size)

                        if image_idx % plot_every_n == 0:
                            cv2.imwrite(join(output_folder, 'raw', '{}_{}.png').format(
                                image_idx, method_name), img)

                        if not args.no_equalize_histogram:
                            img = equalize(img)

                        if image_idx % plot_every_n == 0:
                            cv2.imwrite(join(output_folder, 'processed', '{}_{}.png'.format(
                                image_idx, method_name)), img)

                        img = convert_image(img)

                        if not img.shape == groundtruth_img.shape:
                            print('Inconsistent image sizes. Ignoring image index: {}'.format(image_idx))

                            # fill the metric values and (eventual) preview image with NaNs or zeros
                            for metric_name in metric_names:
                                results[metric_name][method_name].append(np.nan)

                            if image_idx % plot_every_n == 0:
                                previews[method_name].append(
                                    np.zeros_like(groundtruth_img[0, 0]))

                            continue

                        if args.eval_temporal_error:
                            # only used by temporal_consistency_loss which expects a [H x W] image
                            img1 = img[0, 0, :, :]

                        if image_idx % plot_every_n == 0:
                            previews[method_name].append(img[0, 0].copy())

                        for metric_name, metric_func in zip(metric_names, metric_funcs):
                            metric_value = metric_func(img, groundtruth_img)
                            results[metric_name][method_name].append(
                                metric_value)

                        # temporal error
                        if args.eval_temporal_error:
                            img0 = last_images[method_name]
                            if img0 is not None:
                                temporal_error, output_images = temporal_consistency_loss(groundtruth_img0,
                                                                                          groundtruth_img1,
                                                                                          img0, img1,
                                                                                          flow,
                                                                                          output_images=True)
                                temporal_errors[method_name].append(temporal_error)

                                if image_idx % plot_every_n == 0:
                                    cv2.imwrite(join(output_folder, 'temporal_error', '{}_{}_processed0.png'.format(
                                        image_idx, method_name)), convert_to_8bit(img0))
                                    cv2.imwrite(join(output_folder, 'temporal_error', '{}_{}_processed1.png'.format(
                                        image_idx, method_name)), convert_to_8bit(img1))
                                    cv2.imwrite(join(output_folder,
                                                     'temporal_error',
                                                     '{}_{}_processed0_warped_from1.png'.format(
                                                         image_idx, method_name)),
                                                convert_to_8bit(output_images['processed1_warped_to0']))

                            last_images[method_name] = img1.copy()

                    except LookupError as err:  # could not retrieve a matching reconstruction
                        print(err)
                        print('Warning: ignoring image {} for method: {}'.format(
                            image_idx, method_name))
                        # fill the metric values and (eventual) preview image with NaNs or zeros
                        for metric_name, metric_func in zip(metric_names, metric_funcs):
                            results[metric_name][method_name].append(np.nan)

                        if image_idx % plot_every_n == 0:
                            previews[method_name].append(
                                np.zeros_like(groundtruth_img[0, 0]))

            timestamps_file.close()

            # Save results for this dataset
            dataset_results = {}
            for metric_name in metric_names:
                dataset_results[metric_name] = {}
                for method_name in method_names:
                    values = remove_NaNs(results[metric_name][method_name])
                    dataset_results[metric_name][method_name] = np.mean(values)

            # Plot metric statistics
            for metric_name in metric_names:
                plt.figure()
                plt.subplot(121)
                for method_name in method_names:
                    plt.plot(groundtruth_loader.relative_timestamps_restricted,
                             results[metric_name][method_name], label=method_name)
                plt.legend(loc='upper right')
                plt.grid()
                plt.xlabel('Time (s)')
                plt.ylabel(metric_name)
                plt.title(metric_name)

                plt.subplot(122)
                # for the boxplots, we need to filter out the NaN values
                filtered_results = [remove_NaNs(values)
                                    for values in results[metric_name].values()]
                plt.boxplot(filtered_results,
                            labels=results[metric_name].keys())
                plt.ylabel(metric_name)
                plt.savefig(
                    join(output_folder, '{}.pdf'.format(metric_name)), bbox='tight')

            # Save temporal error statistics
            if args.eval_temporal_error:
                dataset_results['temporal_error'] = {}
                for method_name in ['GT', *method_names]:
                    values = remove_NaNs(temporal_errors[method_name])
                    dataset_results['temporal_error'][method_name] = np.mean(values)

            # Save metric + temporal errors for the current dataset
            saved_results[dataset_name] = dataset_results

            # Plot previews
            num_rows = args.num_plots
            # as many columns as number of methods, plus an additional one for the ground truth
            num_cols = len(method_names) + 1
            plt.figure()
            for row_idx in range(num_rows):
                for col_idx, method_name in enumerate(method_names):
                    plt.subplot(num_rows, num_cols, row_idx * num_cols + col_idx + 1)
                    preview = previews[method_name][row_idx]
                    plt.imshow(preview, cmap='gray',
                               interpolation=None, vmin=0.0, vmax=1.0)
                    plt.axis('off')

                    if row_idx == 0:
                        plt.title(method_name)

                preview_groundtruth = previews['groundtruth'][row_idx]
                plt.subplot(num_rows, num_cols, row_idx * num_cols + num_cols)
                plt.imshow(preview_groundtruth, cmap='gray',
                           interpolation=None, vmin=0.0, vmax=1.0)
                plt.axis('off')
                if row_idx == 0:
                    plt.title('Ground truth')
            plt.savefig(join(output_folder, 'previews.pdf'), bbox='tight')

        # Save the final results to a JSON file
        with open(join(base_output_folder, 'results.pkl'), 'wb') as results_file:
            pickle.dump(saved_results, results_file)

    # plt.show()
