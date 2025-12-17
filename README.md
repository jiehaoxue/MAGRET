# MAGRET: A Dataset for Multi-Target Visual Grounding in Remote Sensing Images with Cross-Modal Annotations

![Frame](https://img.shields.io/badge/Frame-pytorch-important.svg)
![license](https://img.shields.io/badge/License-GPLv3-brightgreen.svg)

![RSAM](img/model_framework.png)

This repository contains the official implementation and datasets for the paper "MAGRET: A Dataset for Multi-Target Visual Grounding in Remote Sensing Images with Cross-Modal Annotations".

## Installation

MAGRET needs to be installed first before use. The code requires `python>=3.12`, as well as `torch>=2.6.0` and `torchvision>=0.21.0`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install RSAM on a GPU machine using:

## Getting Started

### Structure

```
rsam/
├── checkpoints                     # Model checkpoints
│   ├── bert-base.bin               # bert_base_uncase
│   └── sam2.1_hiera_base_plus.pt   # hiera_base_plus
├── data/                           # Data directory
│   └── RISORS/                     
│       ├── Annotations/            
│       ├── JPEGImages/            
│       ├── RISORS_test.txt       
│       ├── RISORS_train.txt       
│       └── RISORS_val.txt     
├── rsam/                           # Source code
│   ├── config/                     # Model and Training config files
│   ├── csrc/                       # CUDA code
│   ├── modeling/                   # Model code
│   └── utils/                      # Utility functions
├── training/                       # Train script
├── LICENSE                         # License file
└── README.md                       # This file
```

### Dataset
[MAGRET](https://huggingface.co/datasets/xuejiehao/MAGRET/tree/main)


