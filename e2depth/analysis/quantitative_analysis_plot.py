import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import os
import pickle
import argparse

"""This script will read results exported by the `quantitative_analysis.py` script, and export paper ready tables and figures.

   The input dictionary contains the final results, i.e. the mean metric value for every metric, method and dataset
   It has the following form:
        results =
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
                        "GT": val, "methodA": val, "methodC": val
                        }
                }
            }
"""


def add_mean_entry(results):
    # Add a "mean" entry to the results containing the mean value over all datasets
    list_means = {}
    for dataset_name in results.keys():
        for metric_name in results[dataset_name].keys():
            if metric_name not in list_means:
                list_means[metric_name] = {}

            for method_name in results[dataset_name][metric_name].keys():
                if method_name not in list_means[metric_name]:
                    list_means[metric_name][method_name] = []

                value = results[dataset_name][metric_name][method_name]
                list_means[metric_name][method_name].append(value)

    means = {}
    for metric_name in list_means.keys():
        if metric_name not in means:
            means[metric_name] = {}
        for method_name in list_means[metric_name]:
            means[metric_name][method_name] = np.mean(list_means[metric_name][method_name])

    results['Mean'] = means


def print_table(results, metric_names, method_names):
    # Print metric statistics as a LateX table row
    # format should be as follows:

    # dataset_name && {{methodA & methodB & methodC}} && {{methodA & methodB & methodC}} && {{methodA & methodB & methodC}} \\
    # where the {{ }} blocks correspond respectively to metricA, metricB, metricC

    # Example:
    # dynamic\_6dof && 0.092  & 0.114 	& 0.080  &&  0.229	 	& 0.368	& 0.496 && 0.629  & 0.547 	& 0.427 \\

    for dataset_name in results.keys():
        # Find out which method is the best for each metric,
        # so we can prepend the LateX code of the best method with `\bfseries` in the table
        best_method_for_each_metric = {}
        for metric_name in metric_names:
            # MSE, LPIPS should be minimal, SSIM maximal
            measure_func = np.argmax if metric_name == 'SSIM' else np.argmin
            best_method_for_each_metric[metric_name] = method_names[measure_func(
                [results[dataset_name][metric_name][method_name] for method_name in method_names])]

        print(dataset_name.replace('_', '\_'), end=' ')
        for metric_name in metric_names:
            print('&&', end=' ')
            for method_name in method_names:
                values = results[dataset_name][metric_name][method_name]
                if metric_name == 'temporal_error':
                    values *= 100.0
                # means[metric_name][method_name].append(np.mean(values))
                if method_name == best_method_for_each_metric[metric_name]:
                    print('\\bfseries', end=' ')
                if method_name is not method_names[-1]:
                    print('{:.4f} &'.format(np.mean(values)),
                          end=' ', flush=True)
                else:
                    print('{:.4f} '.format(np.mean(values)),
                          end='', flush=True)
        print('\\\\')


def print_table_ablation_study(results, metric_names, method_names):
    """ Print metric statistics as a LateX table in the "ablation study" format, e.g. like this:

    \begin{tabular}{llll}
    \toprule
    & metricA & metricB & metricC \\
    \midrule
    methodA &     &      &       \\
    methodB &     &      &       \\
    \bottomrule
    \end{tabular}

    """

    dataset_name = 'Mean'
    best_method_for_each_metric = {}
    for metric_name in metric_names:
        # MSE, LPIPS and temporal error should be minimal, SSIM maximal
        measure_func = np.argmax if metric_name == 'SSIM' else np.argmin
        best_method_for_each_metric[metric_name] = method_names[measure_func(
            [results[dataset_name][metric_name][method_name] for method_name in method_names])]

    # table header
    print('\\toprule\n', end='')
    print('&', end=' ')
    for metric_name in metric_names:
        suffix = '\\\\\n' if metric_name is metric_names[-1] else '& '
        print('{} {}'.format(metric_name, suffix), end='')

    print('\\midrule\n', end='', flush=True)
    # table content
    for method_name in method_names:
        print('{} &'.format(method_name), end=' ')
        for metric_name in metric_names:
            value = results[dataset_name][metric_name][method_name]
            if metric_name == 'temporal_error':
                value *= 100.0
            if method_name is best_method_for_each_metric[metric_name]:
                print('\\bfseries', end=' ')
            if metric_name is metric_names[-1]:
                print('{:.4f} \\\\'.format(value), end=' ')
            else:
                print('{:.4f} &'.format(value), end=' ')
        print('\n', end='', flush=True)

    # table bottom
    print('\\bottomrule\n', end='', flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Plot quantitative analysis results loaded from a folder')
    parser.add_argument(
        '--results_folder', required=True, help="Path to folder containing the results")

    args = parser.parse_args()

    with open(os.path.join(args.results_folder, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)

    # Add a "mean" entry to the results containing the mean value over all datasets
    add_mean_entry(results)

    print('Image quality table: \n')
    print_table(results, metric_names=['MSE', 'SSIM', 'LPIPS'], method_names=['HF', 'MR', 'Ours_RT'])
    print()

    print('Temporal error table: \n')
    print_table(results, metric_names=['temporal_error'], method_names=['HF', 'MR', 'Ours_RT', 'GT'])
    print()

    print('Ablation study (effect of temporal loss) table: \n')
    print_table_ablation_study(results, metric_names=['MSE', 'SSIM', 'LPIPS', 'temporal_error'], method_names=['Ours_R', 'Ours_RT'])
    print()

    print('Ablation study (effect of recurrent connection) table: \n')
    print_table_ablation_study(results, metric_names=['MSE', 'SSIM', 'LPIPS', 'temporal_error'], method_names=['Ours_LT', 'Ours_RT'])
    print()

    print('Ablation study (effect of window length: 20 Hz vs. 200 Hz) table: \n')
    print_table_ablation_study(results, metric_names=['MSE', 'SSIM', 'LPIPS'], method_names=['Ours_RNN', 'Ours_RNN_200Hz', 'Ours_RT', 'Ours_RT_200Hz'])
    print()
