# DualFS
## Official implementation of "Functionality Separation: Rethinking Dual-Stream Networks for Class-Incremental Learning"

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/class-incremental-learning/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square&logo=python&color=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)


#### Summary

* [Abstract](#introduction)
* [Getting Started](#getting-started)
* [Download the Datasets](#download-the-datasets)
* [Running Experiments](#running-experiments)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

### Abstract

Class-incremental learning (CIL) aims to enable models to continually learn new classes without forgetting previously learned ones, often struggling with the stability-plasticity dilemma. 
The dual-stream architectures adopt one stream with limited learning ability for storing the learned knowledge, and the other stream with high plasticity for learning new information, achieving success in CIL. %balancing stability and plasticity through parameter isolation. 
However, different methods involve completely different components and the design principles for dual-stream architectures remain unclear. 
In this paper, in contrast to the previous methods that focus on a specific point of dual-stream network design, we conduct a large number of experiments across the board to figure out the principles.
It is found that the key to designing dual-stream methods is making each component complete a specific function, namely functionality separation, instead of placing hope in one component to accomplish multiple functions as in many previous works. 
These crucial functions for dual-stream networks are the widely studied stability-maintaining and plasticity-enhancing, together with the generally overlooked knowledge integration introduced in this work. 
Additionally, despite a large number of investigations on regularization methods, a simple combination of classification loss and distillation loss is sufficient for the regularization of dual-stream networks.
Based on all these experiments and analyses, it is natural to acquire the dual-stream model combined with the compression training paradigm, which effectively separates the functions of stability, plasticity, and the integration of new and old knowledge, achieving the state-of-the-art results on CIFAR-100 and ImageNet-Subset CIL benchmarks.


### Getting Started

###### 1. Clone the repository

```bash
git clone https://github.com/gaoqingqing77/DualFS.git
cd DualFS
```


###### 2. Create environment

Recommended Python version: 3.6+

```bash
pip install torch torchvision numpy tqdm scipy sklearn tensorboardX Pillow==6.2.2
pip install torch torchvision numpy tqdm matplotlib
```


### Download the Datasets
#### CIFAR-100
It will be downloaded automatically by `torchvision` when running the experiments.

#### ImageNet-Subset
We create the ImageNet-Subset following [LUCIR](https://github.com/hshustc/CVPR19_Incremental_Learning).
You may download the dataset using the following links:
- [Download from Google Drive](https://drive.google.com/file/d/1n5Xg7Iye_wkzVKc0MTBao5adhYSUlMCL/view?usp=sharing)
- [Download from 百度网盘](https://pan.baidu.com/s/1MnhITYKUI1i7aRBzsPrCSw) (提取码: 6uj5)

File information:
```
File name: ImageNet-Subset.tar
Size: 15.37 GB
MD5: ab2190e9dac15042a141561b9ba5d6e9
```
You need to untar the downloaded file, and put the folder `seed_1993_subset_100_imagenet` in `class-incremental-learning/adaptive-aggregation-networks/data`.

Please note that the ImageNet-Subset is created from ImageNet. ImageNet is only allowed to be downloaded by researchers for non-commercial research and educational purposes. See the terms of ImageNet [here](https://image-net.org/download.php).

### Running Experiments

```bash
python main.py --nb_cl_fg=50 --nb_cl=10 --branch_mode=dual --branch_1=fixed --branch_2=free --dataset=cifar100
python main.py --nb_cl_fg=50 --nb_cl=5 --branch_mode=dual --branch_1=fixed --branch_2=free --dataset=cifar100
python main.py --nb_cl_fg=50 --nb_cl=2 --branch_mode=dual --branch_1=fixed --branch_2=free --dataset=cifar100
```


### Citation

Please cite our paper if it is helpful to your work:

[//]: # (```bibtex)

[//]: # (@inproceedings{Liu2020AANets,)

[//]: # (  author    = {Liu, Yaoyao and Schiele, Bernt and Sun, Qianru},)

[//]: # (  title     = {Adaptive Aggregation Networks for Class-Incremental Learning},)

[//]: # (  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition &#40;CVPR&#41;},)

[//]: # (  pages     = {2544-2553},)

[//]: # (  year      = {2021})

[//]: # (})

[//]: # (```)

### Acknowledgements

This implementation is primarily built upon the publicly available codebase of:

- [Adaptive Aggregation Networks for Class-Incremental Learning](https://github.com/yaoyao-liu/class-incremental-learning/tree/main/adaptive-aggregation-networks)

Certain components are adapted from the following repositories:

- [Learning a Unified Classifier Incrementally via Rebalancing](https://github.com/hshustc/CVPR19_Incremental_Learning)
- [iCaRL: Incremental Classifier and Representation Learning](https://github.com/srebuffi/iCaRL)

We also refer to the official implementation of:

- [FOSTER: Feature Boosting and Compression for Class-Incremental Learning](https://github.com/G-U-N/ECCV22-FOSTER)

We sincerely thank the authors of these works for making their code publicly available.
