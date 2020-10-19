# Learning Monocular Dense Depth from Events

![Learning Monocular Dense Depth from Events](http://rpg.ifi.uzh.ch/E2DEPTH/E2DEPTH_main_grid.png)

This is the code for the paper **Learning Monocular Dense Depth from Events** by
[Javier Hidalgo-Carrió](https://jhidalgocarrio.github.io), [Daniel Gehrig](https://danielgehrig18.github.io/), and [Davide
Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html):

You can find a pdf of the paper
[here](http://rpg.ifi.uzh.ch/docs/3DV20_Hidalgo.pdf).  If you use any of this
code or our event camera plugin in
[CARLA](https://carla.readthedocs.io/en/latest/ref_sensors/#dvs-camera), please
cite the following publication:

```bibtex
@Article{Hidalgo20threedv,
  author        = {Javier Hidalgo-Carrio, Daniel Gehrig and Davide Scaramuzza},
  title         = {Learning Monocular Dense Depth from Events},
  journal       = {{IEEE} International Conference on 3D Vision.(3DV)},
  url           = {http://rpg.ifi.uzh.ch/docs/3DV20_Hidalgo.pdf},
  year          = 2020
}
```

## Install

Dependencies:

- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.0
- [NumPy](https://www.numpy.org/)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org)

### Install with Anaconda

The installation requires [Anaconda3](https://www.anaconda.com/distribution/).
You can create a new Anaconda environment with the required dependencies as
follows (make sure to adapt the CUDA toolkit version according to your setup):

```bash
conda create -n E2DEPTH
conda activate E2DEPTH
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -c conda-forge opencv
conda install -c conda-forge matplotlib
```

## Run

- Download the pretrained model:

```bash
wget "http://rpg.ifi.uzh.ch/data/E2DEPTH/models/E2DEPTH_si_grad_loss_mixed.pth.tar" -O pretrained/E2DEPTH_si_grad_loss_mixed.pth.tar
```

- Download the test sequence in the DENSE dataset:

```bash
wget "http://rpg.ifi.uzh.ch/data/E2DEPTH/dataset/test_sequence_00_town10.zip" -O data/test_sequence_00_town10.zip
```
- Extract the data sequence:

```bash
unzip -q data/test_sequence_00_town10.zip -d data/test
```

Before running the reconstruction, make sure the conda environment is sourced:

```bash
conda activate E2DEPTH
```

- Run reconstruction:

```bash
python run_depth.py -c pretrained/E2DEPTH_si_grad_loss_mixed.pth.tar \
  -i data/test/events/voxels \
  --output_folder /tmp \
  --save_numpy \
  --show_event \
  --display \
  --save_inv_log \
  --save_color_map
```

## Parameters

Below is a description of the most important parameters:

#### Main parameters

#### Output parameters

- ``--output_folder``: path of the output folder. If not set, the image reconstructions will not be saved to disk.
- ``--dataset_name``: name of the output folder directory (default: 'reconstruction').

#### Display parameters

- ``--display`` (default: False): display the video reconstruction in real-time in an OpenCV window.
- ``--show_events`` (default: False): show the input events side-by-side with the reconstruction. If ``--output_folder`` is set, the previews will also be saved to disk in ``/path/to/output/folder/events``.
- ``--save_inv_log`` (default: False): compute (and then save) the inverse depth log instead of the depth log (default).
- ``--save_color_map`` (default: False): use color conding to display depth. It uses matplotlib 'magma' color map. Grayscale depth otherwise.

#### Display window

You can select between direct or inverse depth, log or lineal, and grayscale or color visualization.

![Display window](doc/img/e2depth_display_window.png)

## DENSE dataset

We provide Depth Estimation oN Synthetic Events (DENSE) Dataset that you can use to train your model.

- [DENSE](http://rpg.ifi.uzh.ch/E2DEPTH.html)

## Event Camera plugin

You can extend DENSE or create your own dataset using our
Event camera plugin. You can have a look [here](https://carla.readthedocs.io/en/latest/ref_sensors/#dvs-camera) for a detailed
documentation.

[![Carla with Events](doc/img/sensor_dvs.gif)](https://carla.readthedocs.io/en/latest/ref_sensors/#dvs-camera)

## Acknowledgements

This code borrows from the following open source projects, whom we would like to thank:

- [pytorch-template](https://github.com/victoresque/pytorch-template)
