#!/usr/bin/env bash

# This script allows to export the inference code for the public repository of E2VID, available at: https://github.com/uzh-rpg/rpg_e2vid

E2VID_FOLDER=$(pwd)

# parse path to the rpg_e2vid folder in which the inference code should be exported
if [ $# -ge 1 ]
then
      OUTPUT_FOLDER=$1
else
      OUTPUT_FOLDER=/home/$USER/rpg_e2vid
fi

echo "Will export inference code to folder: $OUTPUT_FOLDER"

mkdir -p $OUTPUT_FOLDER
cp $E2VID_FOLDER/standalone_reconstruction.py $OUTPUT_FOLDER/run_reconstruction.py
cp $E2VID_FOLDER/image_reconstructor.py $OUTPUT_FOLDER
cp $E2VID_FOLDER/export/README.md $OUTPUT_FOLDER

mkdir -p $OUTPUT_FOLDER/options
cp $E2VID_FOLDER/options/__init__.py $OUTPUT_FOLDER/options
cp $E2VID_FOLDER/options/inference_options.py $OUTPUT_FOLDER/options

mkdir -p $OUTPUT_FOLDER/base
cp $E2VID_FOLDER/base/__init__.py $OUTPUT_FOLDER/base
sed -i '$ d' $OUTPUT_FOLDER/base/__init__.py  # this removes the last line "from .base_trainer import *"
cp $E2VID_FOLDER/base/base_model.py $OUTPUT_FOLDER/base

mkdir -p $OUTPUT_FOLDER/model
cp $E2VID_FOLDER/model/__init__.py $OUTPUT_FOLDER/model
cp $E2VID_FOLDER/model/submodules.py $OUTPUT_FOLDER/model
cp $E2VID_FOLDER/model/unet.py $OUTPUT_FOLDER/model
cp $E2VID_FOLDER/model/model.py $OUTPUT_FOLDER/model

mkdir -p $OUTPUT_FOLDER/utils
cp $E2VID_FOLDER/utils/__init__.py $OUTPUT_FOLDER/utils
cp $E2VID_FOLDER/utils/loading_utils.py $OUTPUT_FOLDER/utils
cp $E2VID_FOLDER/utils/path_utils.py $OUTPUT_FOLDER/utils
cp $E2VID_FOLDER/utils/util.py $OUTPUT_FOLDER/utils
cp $E2VID_FOLDER/utils/timers.py $OUTPUT_FOLDER/utils
cp $E2VID_FOLDER/utils/inference_utils.py $OUTPUT_FOLDER/utils
cp $E2VID_FOLDER/utils/event_readers.py $OUTPUT_FOLDER/utils

mkdir -p $OUTPUT_FOLDER/scripts
cp $E2VID_FOLDER/scripts/extract_events_from_rosbag.py $OUTPUT_FOLDER/scripts
cp $E2VID_FOLDER/scripts/image_folder_to_rosbag.py $OUTPUT_FOLDER/scripts
cp $E2VID_FOLDER/scripts/embed_reconstructed_images_in_rosbag.py $OUTPUT_FOLDER/scripts
cp $E2VID_FOLDER/scripts/subsample_reconstructions.py $OUTPUT_FOLDER/scripts/resample_reconstructions.py

mkdir -p $OUTPUT_FOLDER/pretrained
touch $OUTPUT_FOLDER/pretrained/.gitignore

mkdir -p $OUTPUT_FOLDER/data
touch $OUTPUT_FOLDER/data/.gitignore


echo "Done!"
