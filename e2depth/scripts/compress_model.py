import torch
import argparse
from os.path import join

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Compressing a model by stripping out unnecessary stuff for inference')
    parser.add_argument('-i', '--input_folder', required=True, type=str,
                        help='folder containing the model_best.pth.tar file')

    args = parser.parse_args()

    print('Loading model...')
    model = torch.load(join(args.input_folder, 'model_best.pth.tar'))

    # Only keep the useful information from the full model (network weights, training config, architecture name).
    # Strip out the rest (optimizer state, etc.)
    model_compressed = {'arch': model['arch'], 'state_dict': model['state_dict'], 'model': model['config']['model']}

    print('Saving compressed model...')
    torch.save(model_compressed, join(args.input_folder, 'model_compressed.pth.tar'))
    print('Done!')
