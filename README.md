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

<div>
    <img src="./screenshot.png" />
</div>

<p align='center'><strong>Made by <a href='https://github.com/bkkaggle'>Bilal Khan</a> • https://bkkaggle.github.io</strong></p>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

-   [Installation](#installation)
    -   [Colab](#colab)
    -   [Kaggle](#kaggle)
    -   [GCP](#gcp)
        -   [Vscode remote setup](#vscode-remote-setup)
-   [Documentation](#documentation)
    -   [Generation](#generation)
        -   [`python generate.py`](#python-generatepy)
-   [Contributing](#contributing)
-   [Authors](#authors)
-   [License](#license)
-   [Acknowledgements](#acknowledgements)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Installation on GCP

-   `gcloud compute instances create gans --zone="us-west1-b" --image-family="pytorch-latest-cu100" --image-project=deeplearning-platform-release --maintenance-policy=TERMINATE --accelerator="type=nvidia-tesla-v100,count=1" --metadata="install-nvidia-driver=True" --preemptible --boot-disk-size="100GB" --custom-cpu=8 --custom-memory=16`
-   `gcloud compute ssh gans`
-   `sudo apt-get update`
-   `sudo apt-get upgrade`
-   `git clone https://github.com/bkkaggle/gans.git`
-   `cd gans`
-   `conda env create -f environment.yml`
-   `source activate gans`
-   `pip install future`
-   `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`

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

# Documentation

## U-GAT-IT

-   https://github.com/taki0112/UGATIT

    -   pytorch version is here: https://github.com/znxlwm/UGATIT-pytorch
    -   cat2dog saved model is here: https://github.com/taki0112/UGATIT/issues/51
    -   default saved models are available in repo

    -   my notebook (https://colab.research.google.com/drive/1O2BZE8ptAPE0CODbk7yjBgOuPe9YLQZo)

    -   finetune
        -   cat2dog instructions (https://www.kaggle.com/waifuai/ugatit-cat2dog-pretrained-model)

-   https://github.com/NVlabs/FUNIT
-   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
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

-   _Bilal Khan_ - _Forked the repository and added some features_

# License

This project is licensed under the MIT license - see the [license](LICENSE) file for details

# Acknowledgements

This project contains code from: _list_

This README is based on (https://github.com/bkkaggle/grover) and (https://github.com/rish-16/gpt2client)
