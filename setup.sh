#!/usr/bin/env bash

conda create -n anlp_hw2 python=3.11
conda activate anlp_hw2

# detect the operating system
OS=$(uname)
if [ "$OS" = "Linux" ]; then
  conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
elif [ "$OS" = "Darwin" ]; then
  conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -c pytorch
else
  echo "Unsupported operating system: $OS"
fi

pip install -r requirements.txt
