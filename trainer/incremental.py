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
""" Training code for LUCIR """
import torch
import tqdm
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from utils.misc import *
from utils.process_fp import process_inputs_fp

cur_features = []
ref_features = []
old_scores = []
new_scores = []

def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs

def map_labels(order_list, Y_set):
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y

def BKD(per_cls_weights, pred, soft, T=2):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    soft = soft * per_cls_weights
    soft = soft / soft.sum(1)[:, None]
    return -1*torch.mul(soft, pred).sum()/pred.shape[0]

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1*torch.mul(soft, pred).sum()/pred.shape[0]


def incremental_train_and_eval(the_args, epochs, compression_epochs, fusion_vars, ref_fusion_vars, b1_model, ref_model, b2_model, ref_b2_model, b3_model, ref_b3_model, \
    tg_optimizer, tg_lr_scheduler, fusion_optimizer, fusion_lr_scheduler, compression_optimizer, compression_lr_scheduler, trainloader, testloader, iteration, \
    start_iteration, X_protoset_cumuls, Y_protoset_cumuls, order_list, the_lambda, dist, \
    K, lw_mr, per_cls_weights1, per_cls_weights2, balancedloader, fix_bn=False, weight_per_class=None, device=None):

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set the 1st branch reference model to the evaluation mode
    ref_model.eval()

    # Get the number of old classes
    num_old_classes = ref_model.fc.out_features

    # Get the features from the current and the reference model
    handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
    handle_cur_features = b1_model.fc.register_forward_hook(get_cur_features)
    handle_old_scores_bs = b1_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
    handle_new_scores_bs = b1_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)

    # If the 2nd branch reference is not None, set it to the evaluation mode
    if iteration > start_iteration+1:
        ref_b2_model.eval()
        ref_b3_model.eval()

    for epoch in range(epochs):
        # Start training for the current phase, set the two branch models to the training mode
        b1_model.train()
        b2_model.train()

        # Fix the batch norm parameters according to the config
        if fix_bn:
            for m in b1_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # Set all the losses to zeros
        train_loss = 0
        train_loss0 = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        train_loss5 = 0
        # Set the counters to zeros
        correct = 0
        total = 0
    
        # Learning rate decay
        tg_lr_scheduler.step()
        fusion_lr_scheduler.step()

        # Print the information
        print('\nCurrent phase: %d, Epoch: %d, learning rate: ' % (iteration-start_iteration, epoch), end='')
        print(tg_lr_scheduler.get_lr()[0])

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            # Get a batch of training samples, transfer them to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Clear the gradient of the paramaters for the tg_optimizer
            tg_optimizer.zero_grad()

            # Forward the samples in the deep networks
            outputs, cur_features_new, b1_fp_final, b2_fp_final = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)

            #fe_logits = b1_model.fe_fc(b2_fp_final)###

            if iteration == start_iteration+1:
                ref_outputs, _, _, _ = ref_model(inputs, fp_mode=True)
            else:
                ref_outputs, ref_features_new, _, _ = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, inputs)

            # Loss 0: Standard Cross-Entropy Loss
            #loss0 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

            # Loss 1: Weighted Cross-Entropy Loss
            loss1 = F.cross_entropy(
                    outputs/per_cls_weights1.to(device), targets)

            # Loss 2: Plasticity-Enhancing Loss
            #loss2 = F.cross_entropy(fe_logits, targets)

            # Loss 3: Logits Distillation
            loss3 = the_args.lambda_okd * \
                    _KD_loss(outputs[:, :num_old_classes],
                             ref_outputs, the_args.foster_T)

            # Loss 4: Feature Distillation
            #loss4 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), torch.ones(inputs.shape[0]).to(device)) * the_lambda

            # # Loss 5: Relation Distillation
            # # IRD (current)
            # features1_prev_task = F.normalize(outputs, dim=1)
            # features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), 0.2)
            # logits_mask = torch.scatter(
            #     torch.ones_like(features1_sim),
            #     1,
            #     torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
            #     0
            # )
            #
            # logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
            # features1_sim = features1_sim - logits_max1.detach()
            #
            # row_size = features1_sim.size(0)
            #
            # logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
            #     features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
            #
            # # IRD (past)
            # with torch.no_grad():
            #     features2_sim = torch.div(
            #         torch.matmul(F.normalize(ref_outputs, dim=1), F.normalize(ref_outputs, dim=1).T), 0.01)
            #     logits_max2, _ = torch.max(features2_sim * logits_mask, dim=1, keepdim=True)
            #     features2_sim = features2_sim - logits_max2.detach()
            #     logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
            #         features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
            #
            # loss5 = (-logits2 * torch.log(logits1)).sum(1).mean()

            # Sum up all looses
            loss = loss1 + loss3
            # Backward and update the parameters
            loss.backward()
            tg_optimizer.step()

            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
            #train_loss0 += loss0.item()
            train_loss1 += loss1.item()
            #train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            # train_loss4 += loss4.item()
            # train_loss5 += loss5.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Print the training losses and accuracies
        print('Train set: {}, train loss: {:.4f} accuracy: {:.4f},'.format(len(trainloader), train_loss / (batch_idx + 1), 100. * correct / total))
        print('train loss0: {:.4f}, train loss1: {:.4f}, train loss2: {:.4f},'.format(
            train_loss0 / (batch_idx + 1), train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1)))
        print('train loss3: {:.4f}, train loss4: {:.4f}, train loss5: {:.4f},'.format(
            train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss5 / (batch_idx + 1)))

        # Update the aggregation weights
        b1_model.eval()
        b2_model.eval()
     
        for batch_idx, (inputs, targets) in enumerate(balancedloader):
            if batch_idx <= 500:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _, _, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss.backward()
                fusion_optimizer.step()

        # Running the test for this epoch
        b1_model.eval()
        b2_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _, _, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    print("Removing register forward hook")
    handle_ref_features.remove()
    handle_cur_features.remove()
    handle_old_scores_bs.remove()
    handle_new_scores_bs.remove()

    #Compression stage
    b3_model.eval()

    b1_dict = b1_model.state_dict()
    params_to_copy_dict1 = {}
    excluded_names1 = ['fc.sigma', 'fc.fc1.weight', 'fc.fc2.weight',
                      'fe_fc.sigma', 'fe_fc.fc1.weight', 'fe_fc.fc2.weight']
    for name, param in b1_dict.items():
        if all(name not in excluded_name for excluded_name in excluded_names1):
            params_to_copy_dict1[name] = param
    b3_model.load_state_dict(params_to_copy_dict1, strict=False)
    b3_model.to(device)

    for epoch in range(compression_epochs):
        # Start training compression models for the current phase, set the compression models to the training mode
        b3_model.train()

        # Set all the losses to zeros
        compression_train_loss = 0
        # Set the counters to zeros
        correct = 0
        total = 0

        # Learning rate decay
        compression_lr_scheduler.step()

        # Print the information
        print('\nCurrent phase: %d, Compression Epoch: %d, learning rate: ' % (iteration - start_iteration, epoch), end='')
        print(compression_lr_scheduler.get_lr()[0])

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            # Get a batch of training samples, transfer them to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Clear the gradient of the paramaters for the tg_optimizer
            compression_optimizer.zero_grad()

            # Forward the samples in the deep networks
            dual_outputs, _, _, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
            outputs = b3_model(inputs)

            # Loss: compression distillation loss
            loss = BKD(per_cls_weights2.to(device), outputs, dual_outputs.detach(), 2)

            # Backward and update the parameters
            loss.backward()
            compression_optimizer.step()

            # Record the losses and the number of samples to compute the accuracy
            compression_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Print the training losses and accuracies
        print(
            'CompressionTask Train set: {}, compression train loss: {:.4f} accuracy: {:.4f}'.format(
                len(trainloader), compression_train_loss / (batch_idx + 1), 100. * correct / total))

        # Running the test for this epoch
        b3_model.eval()
        compression_test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = b3_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                compression_test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('CompressionTask Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), compression_test_loss / (batch_idx + 1),
                                                                       100. * correct / total))
    return b1_model, b2_model, b3_model
