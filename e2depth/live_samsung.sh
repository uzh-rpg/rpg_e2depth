#!/bin/bash

additional_args=$@

python live_reconstruction.py -c $PRETRAINED_MODELS/E2VID_lightweight.pth.tar --unsharp_mask_amount=0.3 --use_fp16 --display --flip --gamma=1.1 --output_folder=/tmp/samsung_gen3 $additional_args
