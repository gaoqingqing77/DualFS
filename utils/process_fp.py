##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Max Planck Institute for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2021
##
## Modified by: Qingqing Gao
## Beijing Institute of Artificial Intelligence, Beijing University of Technology
## Modifications: dual-stream network design for functionality separation
##
## The original source code is licensed under the MIT License.
## This modified version is also released under the MIT License.
""" Using the aggregation weights to compute the feature maps from two branches """
import torch
import torch.nn as nn
from utils.misc import *

def process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs, feature_mode=False):

    # The 1st level
    if the_args.dataset == 'cifar100':
        b1_model_group1 = [b1_model.conv1, b1_model.bn1, b1_model.relu, b1_model.layer1]
        b2_model_group1 = [b2_model.conv1, b2_model.bn1, b2_model.relu, b2_model.layer1]
    elif the_args.dataset == 'imagenet_sub' or the_args.dataset == 'imagenet':
        b1_model_group1 = [b1_model.conv1, b1_model.bn1, b1_model.relu, b1_model.maxpool, b1_model.layer1]
        b2_model_group1 = [b2_model.conv1, b2_model.bn1, b2_model.relu, b2_model.maxpool, b2_model.layer1]
    else:
        raise ValueError('Please set correct dataset.')
    b1_model_group1 = nn.Sequential(*b1_model_group1)
    b1_fp1 = b1_model_group1(inputs)
    b2_model_group1 = nn.Sequential(*b2_model_group1)
    b2_fp1 = b2_model_group1(inputs)

    # The 2nd level
    b1_model_group2 = b1_model.layer2
    b1_fp2 = b1_model_group2(b1_fp1)
    b2_model_group2 = b2_model.layer2
    b2_fp2 = b2_model_group2(b2_fp1)

    # The 3rd level
    if the_args.dataset == 'cifar100':
        b1_model_group3 = [b1_model.layer3, b1_model.avgpool]
        b2_model_group3 = [b2_model.layer3, b2_model.avgpool]
    elif the_args.dataset == 'imagenet_sub' or the_args.dataset == 'imagenet':
        b1_model_group3 = b1_model.layer3
        b2_model_group3 = b2_model.layer3
    else:
        raise ValueError('Please set correct dataset.')
    b1_model_group3 = nn.Sequential(*b1_model_group3)
    b1_fp3 = b1_model_group3(b1_fp2)
    b2_model_group3 = nn.Sequential(*b2_model_group3)
    b2_fp3 = b2_model_group3(b2_fp2)

    if the_args.dataset == 'cifar100': 
        b1_fp_final = b1_fp3.view(b1_fp3.size(0), -1)
        b2_fp_final = b2_fp3.view(b2_fp3.size(0), -1)
    elif the_args.dataset == 'imagenet_sub' or the_args.dataset == 'imagenet':
        # The 4th level
        b1_model_group4 = [b1_model.layer4, b1_model.avgpool]
        b1_model_group4 = nn.Sequential(*b1_model_group4)
        b1_fp4 = b1_model_group4(b1_fp3)
        b2_model_group4 = [b2_model.layer4, b2_model.avgpool]
        b2_model_group4 = nn.Sequential(*b2_model_group4)
        b2_fp4 = b2_model_group4(b2_fp3)
        b1_fp_final = b1_fp4.view(b1_fp4.size(0), -1)
        b2_fp_final = b2_fp4.view(b2_fp4.size(0), -1)
    else:
        raise ValueError('Please set correct dataset.')

    fp_final = fusion_vars[2] * b1_fp_final + (1-fusion_vars[2]) * b2_fp_final
    if feature_mode:
        return fp_final
    else:
        outputs = b1_model.fc(fp_final)
        return outputs, fp_final, b1_fp_final, b2_fp_final
