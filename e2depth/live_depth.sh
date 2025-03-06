#!/bin/bash

additional_args=$@

python live_reconstruction.py -c $PRETRAINED_MODELS/E2DEPTH_perceptual.pth.tar --unsharp_mask_amount=0.3 --use_fp16 --display  --output_folder=/tmp/depth_perceptual/ $additional_args
