<h1 align='center'>
    GANs • 🤖📷
</h1>

<h4 align='center'>
    A compilation of GAN demos
</h4>

<p align='center'>
    <a href="https://forthebadge.com">
        <img src="https://forthebadge.com/images/badges/made-with-python.svg" alt="forthebadge">
    </a>
    <a href="https://github.com/prettier/prettier">
        <img src="https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square" alt="code style: prettier" />
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    </a>
    <a href="http://makeapullrequest.com">
        <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome">
    </a>
    <a href="https://github.com/bkkaggle/gans/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/bkkaggle/gans">
    </a>

</p>

<p align='center'>
    <a href='#documentation'>Documentation</a> •
    <a href='#contributing'>Contributing</a> •
    <a href='#authors'>Authors</a> •
    <a href='#license'>License</a>
</p>

<p align='center'><strong>Made by <a href='https://github.com/bkkaggle'>Bilal Khan</a> • https://bkkaggle.github.io</strong></p>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

-   [Installation](#installation)
    -   [Make vm](#make-vm)
    -   [Connect to vm](#connect-to-vm)
    -   [Setup vm](#setup-vm)
    -   [Clone repository](#clone-repository)
    -   [Vscode remote editing](#vscode-remote-editing)
    -   [Create environment](#create-environment)
    -   [Install pytorch](#install-pytorch)
-   [install apex](#install-apex)
    -   [Setup kaggle's api](#setup-kaggles-api)
    -   [Download datasets from google drive](#download-datasets-from-google-drive)
-   [Documentation](#documentation)
    -   [U-GAT-IT](#u-gat-it)
        -   [Colab](#colab)
        -   [GCP (Pytorch version)](#gcp-pytorch-version)
    -   [FUNIT](#funit)
        -   [Colab](#colab-1)
        -   [GCP](#gcp)
            -   [Training](#training)
            -   [Testing](#testing)
    -   [CycleGAN](#cyclegan)
        -   [Colab](#colab-2)
        -   [GCP](#gcp-1)
            -   [Datasets](#datasets)
            -   [Pretrained models](#pretrained-models)
            -   [Training](#training-1)
            -   [Testing](#testing-1)
    -   [Pix2Pix](#pix2pix)
        -   [Colab](#colab-3)
        -   [GCP](#gcp-2)
            -   [Datasets](#datasets-1)
            -   [Pretrained models](#pretrained-models-1)
            -   [Training](#training-2)
            -   [Testing](#testing-2)
-   [Contributing](#contributing)
-   [Authors](#authors)
-   [License](#license)
-   [Acknowledgements](#acknowledgements)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Installation

### Make vm

-   `gcloud compute instances create gans --zone="us-west1-b" --image-family="pytorch-latest-cu100" --image-project=deeplearning-platform-release --maintenance-policy=TERMINATE --accelerator="type=nvidia-tesla-v100,count=1" --metadata="install-nvidia-driver=True" --preemptible --boot-disk-size="100GB" --custom-cpu=8 --custom-memory=16`

### Connect to vm

-   `gcloud compute ssh gans`

### Setup vm

-   `sudo apt-get update`
-   `sudo apt-get upgrade`

-   `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
-   `bash Miniconda3-latest-Linux-x86_64.sh`
-   follow instruction and choose to add to your path

### Clone repository

-   `git clone https://github.com/bkkaggle/gans.git`

### Vscode remote editing

-   install https://marketplace.visualstudio.com/items?itemName=rafaelmaiolla.remote-vscode on vscode
-   cmd-shift-p and type `Remote: Start Server`
-   `gcloud compute ssh gans --ssh-flag="-R 52698:localhost:52698"`
-   run `sudo apt -y install ruby && sudo gem install rmate` on vm
-   to edit a file run `rmate path/to/file`

### Create environment

-   `conda env create -f environment.yml`
-   `conda activate gans`
-   `pip install future`

### Install pytorch

-   get cuda version `nvcc --version`
-   install pytorch `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`

## install apex

-   `git clone https://github.com/NVIDIA/apex`
-   `cd apex`
-   `pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`

### Setup kaggle's api

-   `vim ~/.kaggle/kaggle.json`
-   paste in api key `{"username":"[USERNAME]","key":"[API_KEY]"}`

### Download datasets from google drive

-   `wget https://github.com/gdrive-org/gdrive/releases/download/2.1.0/gdrive-linux-x64`
-   `mv gdrive-linux-x64 gdrive`
-   `chmod +x gdrive`
-   authenticate with `./gdrive about`

# Documentation

## U-GAT-IT

There is a [Tensorflow version](https://github.com/taki0112/UGATIT) and a [Pytorch version](https://github.com/znxlwm/UGATIT-pytorch)

### Colab

There is no currently available pretrained model for pytorch, so the easiest way to get started is to use my colab notebook

<a href="https://colab.research.google.com/github/bkkaggle/gans/blob/master/UGATIT.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
</a>

### GCP (Pytorch version)

Train on your own dataset as shown in the [repository](https://github.com/znxlwm/UGATIT-pytorch#usage). There currently aren't any pretrained pytorch models.

-   Clone the pytorch repo `git clone https://github.com/znxlwm/UGATIT-pytorch.git`

-   Download the dataset of your choice

    -   cat2dog `kaggle datasets download -d waifuai/ugatit-cat2dog-pretrained-model`

-   Unzip the dataset
-   Create a dataset folder under `/dataset` for your dataset (If you're using a pretrained model, the name of the folder must match the name of the saved checkpoint, e.g. cat2dog).
-   Create subfolders `testA`, `testB`, `trainA`, and `trainB` under your dataset's folder. Place any images you want to transform from a to b (cat2dog) in the `testA` folder, images you want to transform from b to a (dog2cat) in the `testB` folder, and do the same for the `trainA` and `trainB` folders if you want to train your own GAN.
-   Train the model `python main.py --dataset cat2dog --phase train`. Add `--light=True` if you get OOM errors.

## FUNIT

I'm using the official [pytorch](https://github.com/NVlabs/FUNIT) implementation.

### Colab

<a href="https://colab.research.google.com/github/bkkaggle/gans/blob/master/FUNIT.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
</a>

### GCP

Train on your own dataset, or use a pretrained model

Follow the instructions in the [README](https://github.com/NVlabs/FUNIT):

-   `git clone https://github.com/NVlabs/FUNIT.git`

#### Training

Use the authors animal face dataset and follow the instructions [here](https://github.com/NVlabs/FUNIT#animal-face-dataset) or substitute it with your own.

#### Testing

Download the pretrained model from Google drive

-   `./gdrive download 1CsmSSWyMngtOLUL5lI-sEHVWc2gdJpF9`
-   `mkdir pretrained`
-   `mv pretrained.tar.gz pretrained/`
-   `tar -xf pretrained.tar.gz`

Run the model as shown in the [README](https://github.com/NVlabs/FUNIT):

-   `python test_k_shot.py --config configs/funit_animals.yaml --ckpt pretrained/animal149_gen.pt --input images/input_content.jpg --class_image_folder images/n02138411 --output ./output.jpg`

Point `--input` to the image you want to transform, `--class_image_folder` to a folder with a few images of the class you want to transform the input image to, and `--output` to the path for the output image.

## CycleGAN

I'm using [this](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository's CycleGAN implementation. It also contains instructions for how to use CycleGAN on multiple datasets.

### Colab

<a href="https://colab.research.google.com/github/bkkaggle/gans/blob/master/CycleGAN.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
</a>

### GCP

To install, follow the instructions in the [repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), or use my copied instructions here.

-   `git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix`

This repository uses a few custom pip packages, install them with

-   `pip install -r requirements.txt`

#### Datasets

Download one of the official datasets with:

-   `bash ./datasets/download_cyclegan_dataset.sh [apple2orange, orange2apple, summer2winter_yosemite, winter2summer_yosemite, horse2zebra, zebra2horse, monet2photo, style_monet, style_cezanne, style_ukiyoe, style_vangogh, sat2map, map2sat, cityscapes_photo2label, cityscapes_label2photo, facades_photo2label, facades_label2photo, iphone2dslr_flower]`

Or use your own dataset by creating the appropriate folders and adding in the images.

-   Create a dataset folder under `/dataset` for your dataset.
-   Create subfolders `testA`, `testB`, `trainA`, and `trainB` under your dataset's folder. Place any images you want to transform from a to b (cat2dog) in the `testA` folder, images you want to transform from b to a (dog2cat) in the `testB` folder, and do the same for the `trainA` and `trainB` folders.

#### Pretrained models

Download one of the official pretrained models with:

-   `bash ./scripts/download_cyclegan_model.sh [apple2orange, orange2apple, summer2winter_yosemite, winter2summer_yosemite, horse2zebra, zebra2horse, monet2photo, style_monet, style_cezanne, style_ukiyoe, style_vangogh, sat2map, map2sat, cityscapes_photo2label, cityscapes_label2photo, facades_photo2label, facades_label2photo, iphone2dslr_flower]`

Or add your own pretrained model to `./checkpoints/{NAME}_pretrained/latest_net_G.pt`

#### Training

-   `python train.py --dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan`

Change the `--dataroot` and `--name` to your own dataset's path and model's name. Use `--gpu_ids 0,1,..` to train on multiple GPUs and `--batch_size` to change the batch size. I've found that a batch size of 16 fits onto 4 V100s and can finish training an epoch in ~90s.

Once your model has trained, copy over the last checkpoint to a format that the testing model can automatically detect:

Use `cp ./checkpoints/horse2zebra/latest_net_G_A.pth ./checkpoints/horse2zebra/latest_net_G.pth` if you want to transform images from class A to class B and `cp ./checkpoints/horse2zebra/latest_net_G_B.pth ./checkpoints/horse2zebra/latest_net_G.pth` if you want to transform images from class B to class A.

#### Testing

-   `python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout`

Change the `--dataroot` and `--name` to be consistent with your trained model's configuration.

> from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix:
> The option --model test is used for generating results of CycleGAN only for one side. This option will automatically set --dataset_mode single, which only loads the images from one set. On the contrary, using --model cycle_gan requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at ./results/. Use --results_dir {directory_path_to_save_result} to specify the results directory.

> For your own experiments, you might want to specify --netG, --norm, --no_dropout to match the generator architecture of the trained model.

## Pix2Pix

### Colab

<a href="https://colab.research.google.com/github/bkkaggle/gans/blob/master/Pix2Pix.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
</a>

### GCP

To install, follow the instructions in the [repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), or use my copied instructions here.

-   `git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix`

This repository uses a few custom pip packages, install them with

-   `pip install -r requirements.txt`

#### Datasets

Download one of the official datasets with:

-   `bash ./datasets/download_pix2pix_dataset.sh [cityscapes, night2day, edges2handbags, edges2shoes, facades, maps]`

Or use your own dataset by creating the appropriate folders and adding in the images. Follow the instructions [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md#pix2pix-datasets).

#### Pretrained models

Download one of the official pretrained models with:

-   `bash ./scripts/download_pix2pix_model.sh [edges2shoes, sat2map, map2sat, facades_label2photo, and day2night]`

Or add your own pretrained model to `./checkpoints/{NAME}_pretrained/latest_net_G.pt`

#### Training

-   `python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA`

Change the `--dataroot` and `--name` to your own dataset's path and model's name. Use `--gpu_ids 0,1,..` to train on multiple GPUs and `--batch_size` to change the batch size. Add `--direction BtoA` if you want to train a model to transfrom from class B to A.

#### Testing

-   `python test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name facades_label2photo_pretrained`

Change the `--dataroot`, `--name`, and `--direction` to be consistent with your trained model's configuration and how you want to transform images.

> from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix:
> Note that we specified --direction BtoA as Facades dataset's A to B direction is photos to labels.

> If you would like to apply a pre-trained model to a collection of input images (rather than image pairs), please use --model test option. See ./scripts/test_single.sh for how to apply a model to Facade label maps (stored in the directory facades/testB).

> See a list of currently available models at ./scripts/download_pix2pix_model.sh

-   https://github.com/ShenYujun/InterFaceGAN
-   https://github.com/ali-design/gan_steerability
-   https://github.com/CSAILVision/gandissect
-   https://github.com/zllrunning/video-object-removal
-   https://github.com/Puzer/stylegan-encoder
-   https://github.com/facebookresearch/pytorch_GAN_zoo
-   https://github.com/NVlabs/stylegan
-   https://github.com/shaoanlu/fewshot-face-translation-GAN
-   https://github.com/SummitKwan/transparent_latent_gan

# Contributing

This repository is still a work in progress, so if you find a bug, think there is something missing, or have any suggestions for new features, feel free to open an issue or a pull request. Feel free to use the library or code from it in your own projects, and if you feel that some code used in this project hasn't been properly accredited, please open an issue.

# Authors

-   _Bilal Khan_ - _Initial work_

# License

This project is licensed under the MIT license - see the [license](LICENSE) file for details

# Acknowledgements

This project contains code from: (https://github.com/taki0112/UGATIT), (https://github.com/NVlabs/FUNIT), (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

This README is based on (https://github.com/bkkaggle/grover) and (https://github.com/rish-16/gpt2client)
