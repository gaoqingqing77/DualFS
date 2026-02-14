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
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Class-incremental learning trainer. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import time
import os
import os.path as osp
import sys
import copy
import argparse
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle
import math
import utils.misc
import models.modified_resnet_cifar as modified_resnet_cifar
import models.modified_resnetmtl_cifar as modified_resnetmtl_cifar
import models.modified_resnet as modified_resnet
import models.modified_resnetmtl as modified_resnetmtl
import models.modified_linear as modified_linear
from utils.imagenet.utils_dataset import split_images_labels
from utils.imagenet.utils_dataset import merge_images_labels
from utils.incremental.compute_accuracy import compute_accuracy
from trainer.incremental import incremental_train_and_eval as incremental_train_and_eval
from trainer.zeroth_phase import incremental_train_and_eval_zeroth_phase as incremental_train_and_eval_zeroth_phase
from utils.misc import process_mnemonics
from trainer.base_trainer import BaseTrainer
import warnings
warnings.filterwarnings('ignore')

class Trainer(BaseTrainer):
    def train(self):
        """The class that contains the code for the class-incremental system.
        This trianer is based on the base_trainer.py in the same folder.
        If you hope to find the source code of the functions used in this trainer, you may find them in base_trainer.py.
        """

        # Set tensorboard recorder
        self.train_writer = SummaryWriter(comment=self.save_path)

        # Initial the array to store the accuracies for each phase
        top1_acc_list_cumul = np.zeros((int(self.args.num_classes/self.args.nb_cl), 3, 1))
        top1_acc_list_ori = np.zeros((int(self.args.num_classes/self.args.nb_cl), 3, 1))

        # Load the training and test samples from the dataset
        X_train_total, Y_train_total, X_valid_total, Y_valid_total = self.set_dataset()

        # Initialize the aggregation weights
        self.init_fusion_vars()       

        # Initialize the class order
        order, order_list = self.init_class_order()

        # np.random.seed(None) # edit by lxy

        # Set empty lists for the data    
        X_valid_cumuls    = []
        X_protoset_cumuls = []
        X_train_cumuls    = []
        Y_valid_cumuls    = []
        Y_protoset_cumuls = []
        Y_train_cumuls    = []

        # Initialize the prototypes
        alpha_dr_herding, prototypes = self.init_prototypes(self.dictionary_size, order, X_train_total, Y_train_total)
        # Set the starting iteration
        # We start training the class-incremental learning system from e.g., 50 classes to provide a good initial encoder
        start_iter = int(self.args.nb_cl_fg/self.args.nb_cl)-1

        # Set the models and some parameter to None
        # These models and parameters will be assigned in the following phases
        b1_model = None
        ref_model = None
        b2_model = None
        ref_b2_model = None
        b3_model = None
        ref_b3_model = None
        the_lambda_mult = None
        cls_num_list = [self.args.nb_protos] * (start_iter + 1) * self.args.nb_cl

        # Initialize variables, which will be used to calculate stability and plasticity.
        total_old_class_accuracy = 0
        total_old_class_count = 0
        total_new_class_accuracy = 0
        total_new_class_count = 0
        stability_list = []
        plasticity_list = []

        for iteration in range(start_iter, int(self.args.num_classes/self.args.nb_cl)):
            ### Setting loss hyperparameters
            if iteration > start_iter:
                cls_num_list = [self.args.nb_protos] * len(cls_num_list)
                cls_num_list += [self.dictionary_size] * self.args.nb_cl

                if len(self.args.beta1) > 1:
                    beta1 = self.args.beta1[iteration - start_iter - 1]
                else:
                    beta1 = self.args.beta1[0]
                effective_num1 = 1.0 - np.power(beta1, cls_num_list)
                per_cls_weights1 = (1.0 - beta1) / np.array(effective_num1)
                per_cls_weights1 = per_cls_weights1 / \
                                  np.sum(per_cls_weights1) * len(cls_num_list)
                per_cls_weights1 = torch.FloatTensor(per_cls_weights1)
                # Update compression hyperparameters
                if len(self.args.beta2) > 1:
                    beta2 = self.args.beta2[iteration-start_iter - 1]
                else:
                    beta2 = self.args.beta2[0]
                effective_num2 = 1.0 - np.power(beta2, cls_num_list)
                per_cls_weights2 = (1.0 - beta2) / np.array(effective_num2)
                per_cls_weights2 = per_cls_weights2 / \
                                  np.sum(per_cls_weights2) * len(cls_num_list)
                per_cls_weights2 = torch.FloatTensor(per_cls_weights2)

            ### Initialize models for the current phase
            b1_model, b2_model, b3_model, ref_model, ref_b2_model, ref_b3_model, lambda_mult, cur_lambda, last_iter = self.init_current_phase_model(iteration, start_iter, b1_model, b2_model, b3_model)

            ### Initialize datasets for the current phase
            if iteration == start_iter:
                indices_train_10, X_valid_cumul, X_train_cumul, Y_valid_cumul, Y_train_cumul, \
                    X_train_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls, X_valid_cumuls, Y_valid_cumuls, \
                    X_train, map_Y_train, map_Y_valid_cumul, X_valid_ori, Y_valid_ori = \
                    self.init_current_phase_dataset(iteration, \
                    start_iter, last_iter, order, order_list, X_train_total, Y_train_total, X_valid_total, Y_valid_total, \
                    X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls)
            else:
                indices_train_10, X_valid_cumul, X_train_cumul, Y_valid_cumul, Y_train_cumul, \
                    X_train_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls, X_valid_cumuls, Y_valid_cumuls, \
                    X_train, map_Y_train, map_Y_valid_cumul, X_protoset, Y_protoset = \
                    self.init_current_phase_dataset(iteration, \
                    start_iter, last_iter, order, order_list, X_train_total, Y_train_total, X_valid_total, Y_valid_total, \
                    X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls)                

            is_start_iteration = (iteration == start_iter)

            # Imprint weights
            if iteration > start_iter:
                b1_model, b3_model = self.imprint_weights(b1_model, b2_model, b3_model, iteration, is_start_iteration, X_train, map_Y_train, self.dictionary_size)

            # Update training and test dataloader
            trainloader, testloader = self.update_train_and_valid_loader(X_train, map_Y_train, X_valid_cumul, map_Y_valid_cumul, \
                iteration, start_iter)

            # Set the names for the checkpoints
            ckp_name = osp.join(self.save_path, 'iter_{}_b1.pth'.format(iteration))
            ckp_name_b2 = osp.join(self.save_path, 'iter_{}_b2.pth'.format(iteration))
            ckp_name_b3 = osp.join(self.save_path, 'iter_{}_b3.pth'.format(iteration))
            print('Check point name: ', ckp_name)

            if iteration==start_iter and self.args.resume_fg:
                # Resume the 0-th phase model according to the config
                tg_dict = torch.load(self.args.ckpt_dir_fg)
                b1_model.load_state_dict(tg_dict)
                b1_model.to(self.device)
            elif self.args.resume and os.path.exists(ckp_name):
                # Resume other models according to the config
                tg_dict = torch.load(ckp_name)
                b1_model.load_state_dict(tg_dict)
                b1_model.to(self.device)
                if os.path.exists(ckp_name_b2):
                    b2_dict = torch.load(ckp_name_b2)
                    b2_model.load_state_dict(b2_dict)
                    b2_model.to(self.device)
                if os.path.exists(ckp_name_b3):
                    b3_dict = torch.load(ckp_name_b3)
                    b3_model.load_state_dict(b3_dict)
                    b3_model.to(self.device)
            else:
                # Start training (if we don't resume the models from the checkppoints)
    
                # Set the optimizer
                tg_optimizer, tg_lr_scheduler, fusion_optimizer, fusion_lr_scheduler, compression_optimizer, compression_lr_scheduler = self.set_optimizer(iteration, \
                    start_iter, b1_model, ref_model, b2_model, ref_b2_model, b3_model, ref_b3_model)

                if iteration > start_iter:
                    # Training the class-incremental learning system from the 1st phase

                    # Set the balanced dataloader
                    balancedloader = self.gen_balanced_loader(X_train_total, Y_train_total, indices_train_10, X_protoset, Y_protoset, order_list)

                    # Training the model for different baselines        
                    if self.args.baseline == 'lucir':
                        b1_model, b2_model, b3_model = incremental_train_and_eval(self.args, self.args.epochs, self.args.compression_epochs, self.fusion_vars, \
                            self.ref_fusion_vars, b1_model, ref_model, b2_model, ref_b2_model, b3_model, ref_b3_model, tg_optimizer, tg_lr_scheduler, \
                            fusion_optimizer, fusion_lr_scheduler, compression_optimizer, compression_lr_scheduler, trainloader, testloader, iteration, start_iter, \
                            X_protoset_cumuls, Y_protoset_cumuls, order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr, per_cls_weights1, per_cls_weights2, balancedloader)
                    else:
                        raise ValueError('Please set the correct baseline.')       
                else:         
                    # Training the class-incremental learning system from the 0th phase           
                    b1_model = incremental_train_and_eval_zeroth_phase(self.args, self.args.init_epochs, b1_model, \
                        ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iter, \
                        cur_lambda, self.args.dist, self.args.K, self.args.lw_mr)

                # Save the model dictionary
                torch.save(b1_model.state_dict(), ckp_name)
                if b2_model is not None:
                    torch.save(b2_model.state_dict(), ckp_name_b2)
                if b3_model is not None:
                    torch.save(b3_model.state_dict(), ckp_name_b3)

            # Select the exemplars according to the current model
            X_protoset_cumuls, Y_protoset_cumuls, class_means, alpha_dr_herding = self.set_exemplar_set(b1_model, b2_model, \
                is_start_iteration, iteration, last_iter, order, alpha_dr_herding, prototypes)
            
            # Compute the accuracies for current phase
            top1_acc_list_ori, top1_acc_list_cumul, each_class_acc = self.compute_acc(class_means, order, order_list, b1_model, b2_model, X_protoset_cumuls, Y_protoset_cumuls, \
                X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul, iteration, is_start_iteration, top1_acc_list_ori, top1_acc_list_cumul)

            ### Print the accuracy of each phase
            acc_fc = []
            acc_icarl = []
            for cumul_acc_fc in top1_acc_list_cumul[start_iter:,0]:
                acc_fc.append(cumul_acc_fc.item())
            for cumul_acc_icarl in top1_acc_list_cumul[start_iter:,1]:
                acc_icarl.append(cumul_acc_icarl.item())
            print('Print the information of each phase...')
            acc_fc_formatted = ["{:.2f}".format(item) for item in acc_fc]
            print("  Accuracy (FC)    :  ", ", ".join(acc_fc_formatted), "%")
            acc_icarl_formatted = ["{:.2f}".format(item) for item in acc_icarl]
            print("  Accuracy (Proto) :  ", ", ".join(acc_icarl_formatted), "%")

            #Update variables, which will be used to calculate stability and plasticity
            if iteration > start_iter:
                old_class_accuracy_sum = each_class_acc[:-self.args.nb_cl].sum()
                old_class_count = iteration * self.args.nb_cl

                total_old_class_accuracy += old_class_accuracy_sum
                total_old_class_count += old_class_count

                new_class_accuracy_sum = each_class_acc[-self.args.nb_cl:].sum()
                new_class_count = self.args.nb_cl

                total_new_class_accuracy += new_class_accuracy_sum
                total_new_class_count += new_class_count

                stability = total_old_class_accuracy / total_old_class_count if total_old_class_count != 0 else 0
                plasticity = total_new_class_accuracy / total_new_class_count if total_new_class_count != 0 else 0

                stability_list.append(stability)
                plasticity_list.append(plasticity)

                formatted_stability_list = ["{:.2f}".format(item) for item in stability_list]
                print("  Stability values :  ", ", ".join(formatted_stability_list), "%")
                formatted_plasticity_list = ["{:.2f}".format(item) for item in plasticity_list]
                print("  Plasticity values:  ", ", ".join(formatted_plasticity_list), "%")

            # Compute the average accuracy
            num_of_testing = iteration - start_iter + 1  # 任务(phase)的数量
            avg_cumul_acc_fc = np.sum(top1_acc_list_cumul[start_iter:, 0]) / num_of_testing  # 精度总和/任务数量=平均精度
            avg_cumul_acc_icarl = np.sum(top1_acc_list_cumul[start_iter:, 1]) / num_of_testing
            print('Computing average accuracy...')
            print("  Average accuracy (FC)         :\t\t{:.2f} %".format(avg_cumul_acc_fc))
            print("  Average accuracy (Proto)      :\t\t{:.2f} %".format(avg_cumul_acc_icarl), "\n")

            # Write the results to the tensorboard
            self.train_writer.add_scalar('avg_acc/fc', float(avg_cumul_acc_fc), iteration)
            self.train_writer.add_scalar('avg_acc/proto', float(avg_cumul_acc_icarl), iteration)

        # Calculate stability and plasticity
        print('Computing Stability and Plasticity...')
        stability = total_old_class_accuracy / total_old_class_count
        plasticity = total_new_class_accuracy / total_new_class_count
        print(f"  Total Stability  :  {stability:.2f} %")
        print(f"  Total Plasticity :  {plasticity:.2f} %")

        # Save the results and close the tensorboard writer
        torch.save(top1_acc_list_ori, osp.join(self.save_path, 'acc_list_ori.pth'))
        torch.save(top1_acc_list_cumul, osp.join(self.save_path, 'acc_list_cumul.pth'))
        self.train_writer.close()
