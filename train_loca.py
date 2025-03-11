import dataset_loca as dataset
import utils
from utils import EarlyStopping, LRScheduler
import os
import pandas as pd
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PET lymphoma classification with Localization')

# I/O PARAMS
parser.add_argument('--output', type=str, default='results',
                    help='name of output folder (default: "results")')

# MODEL PARAMS
parser.add_argument('--normalize', action='store_true', default=False,
                    help='normalize images')
parser.add_argument('--checkpoint', default='', type=str,
                    help='model checkpoint if any (default: none)')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume from checkpoint')

# OPTIMIZATION PARAMS
parser.add_argument('--optimizer', default='sgd', type=str,
                    help='The optimizer to use (default: sgd)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--lr_anneal', type=int, default=15,
                    help='period for lr annealing (default: 15). Only works for SGD')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')

# TRAINING PARAMS
parser.add_argument('--split_index', default=0, type=int, metavar='INT',
                    choices=list(range(0, 20)), help='which split index (default: 0)')
parser.add_argument('--run', default=1, type=int, metavar='INT',
                    help='repetition run with same settings (default: 1)')
parser.add_argument('--batch_size', type=int, default=50,
                    help='how many images to sample per slide (default: 50)')
parser.add_argument('--nepochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 10)')
parser.add_argument('--augm', default=0, type=int,
                    choices=[0, 1, 2, 3, 12, 4, 5, 14, 34, 45],
                    help='augmentation procedure (default: 0)')
parser.add_argument('--balance', action='store_true', default=True,
                    help='balance dataset (balance loss)')
parser.add_argument('--lr_scheduler', action='store_true', default=False,
                    help='decrease LR on plateau')
parser.add_argument('--early_stopping', action='store_true', default=False,
                    help='use early stopping')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='if set, only fine tune the classifier (fc) and freeze the backbone')

def validate_loss(loader, model, criterion_class, criterion_loc, lambda_weight):
    """Compute average combined loss over the validation set."""
    model.eval()
    running_loss = 0.
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            # Expect target to be a tuple: (target_class, target_loc)
            target_class, target_loc = target
            target_class_1hot = F.one_hot(target_class.long(), num_classes=2).cuda()
            target_loc = target_loc.long().cuda()
            output_class, output_loc = model(input)
            loss_class = criterion_class(output_class, target_class_1hot.float())
            loss_loc = criterion_loc(output_loc, target_loc)
            loss = loss_class + lambda_weight * loss_loc
            running_loss += loss.item() * input.size(0)
    return running_loss / len(loader.dataset)

def main():
    global args
    args = parser.parse_args()
    print(args)
    
    # Create output directory if it does not exist.
    if not os.path.isdir(args.output):
        try:
            os.mkdir(args.output)
        except OSError:
            print('Creation of the output directory "{}" failed.'.format(args.output))
        else:
            print('Successfully created the output directory "{}".'.format(args.output))
    
    # Always load the multitask model.
    model = utils.get_model_multitask()
    
    if args.checkpoint:
        ch = torch.load(args.checkpoint)
        model_dict = model.state_dict()
        if 'state_dict' in ch:
            pretrained_dict = {k: v for k, v in ch['state_dict'].items() if k in model_dict}
        else:
            pretrained_dict = {k: v for k, v in ch.items() if k in model_dict}
        print('Loaded [{}/{}] keys from checkpoint'.format(len(pretrained_dict), len(model_dict)))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    if args.resume:
        resume_path = os.path.join(args.output, 'checkpoint_split' + str(args.split_index) +
                                   '_run' + str(args.run) + '.pth')
        with torch.serialization.safe_globals(["numpy._core.multiarray.scalar"]):
            ch = torch.load(resume_path, weights_only=False)
        model_dict = model.state_dict()
        if 'state_dict' in ch:
            pretrained_dict = {k: v for k, v in ch['state_dict'].items() if k in model_dict}
        else:
            pretrained_dict = {k: v for k, v in ch.items() if k in model_dict}
        print('Loaded [{}/{}] keys from checkpoint'.format(len(pretrained_dict), len(model_dict)))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    # Fine-tuning: freeze all layers except the classifier (if desired)
    if args.finetune:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze both classifier heads
        for param in model.fc.parameters():
            param.requires_grad = True
        for param in model.fc_loc.parameters():
            param.requires_grad = True
        print("Fine-tuning mode enabled: only training the classifier layers.")
    else:
        print("Training all layers.")
    
    # Create optimizer for parameters with requires_grad=True.
    optimizer = utils.create_optimizer(
        list(filter(lambda p: p.requires_grad, model.parameters())),
        args.optimizer, args.lr, args.momentum, args.wd
    )
    
    cudnn.benchmark = True
    
    # Augmentations
    flipHorVer = dataset.RandomFlip()
    flipLR = dataset.RandomFlipLeftRight()
    rot90 = dataset.RandomRot90()
    scale = dataset.RandomScale()
    noise = dataset.RandomNoise()
    if args.augm == 0:
        transform = None
    elif args.augm == 1:
        transform = transforms.Compose([flipHorVer])
    elif args.augm == 2:
        transform = transforms.Compose([rot90])
    elif args.augm == 3:
        transform = transforms.Compose([flipLR])
    elif args.augm == 12:
        transform = transforms.Compose([flipHorVer, rot90])
    elif args.augm == 4:
        transform = transforms.Compose([scale])
    elif args.augm == 5:
        transform = transforms.Compose([noise])
    elif args.augm == 14:
        transform = transforms.Compose([flipHorVer, scale])
    elif args.augm == 34:
        transform = transforms.Compose([flipLR, scale])
    elif args.augm == 45:
        transform = transforms.Compose([scale, noise])
    
    # Set datasets using the new helper function from dataset_withLoca.
    train_dset, trainval_dset, val_dset, _, balance_weight_neg_pos = dataset.get_datasets_singleview_withLoca(
        transform, args.normalize, args.balance, args.split_index)
    print('Datasets train:{}, val:{}'.format(len(train_dset.df), len(val_dset.df)))
    print(f"Weight of each class, no tumor: {balance_weight_neg_pos[0]}, tumor: {balance_weight_neg_pos[1]}")
    
    # Set a weighting factor to balance the losses.
    lambda_weight = 1.0
    
    # Set loss criterion for the binary classification branch (with optional balancing)
    if args.balance:
        w = torch.Tensor(balance_weight_neg_pos)
        print('Balance loss with weights:', balance_weight_neg_pos)
        criterion_class = nn.BCEWithLogitsLoss(pos_weight=w).cuda()
    else:
        criterion_class = nn.BCEWithLogitsLoss().cuda()
    
    # Set loss criterion for the localization branch (51 classes)
    criterion_loc = nn.CrossEntropyLoss().cuda()
    
    # Early stopping and learning rate scheduler setup.
    lr_scheduler = None
    early_stopping = None
    if args.lr_scheduler:
        print('INFO: Initializing learning rate scheduler')
        lr_scheduler = LRScheduler(optimizer)
        if args.resume and 'lr_scheduler' in ch:
            lr_scheduler.lr_scheduler.load_state_dict(ch['lr_scheduler'])
            print('Loaded lr_scheduler state')
    if args.early_stopping:
        print('INFO: Initializing early stopping')
        early_stopping = EarlyStopping()
        if args.resume and 'early_stopping' in ch:
            early_stopping.best_loss = ch['early_stopping']['best_loss']
            early_stopping.counter = ch['early_stopping']['counter']
            early_stopping.min_delta = ch['early_stopping']['min_delta']
            early_stopping.patience = ch['early_stopping']['patience']
            early_stopping.early_stop = ch['early_stopping']['early_stop']
            print('Loaded early_stopping state')
    
    # Set data loaders.
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    trainval_loader = torch.utils.data.DataLoader(trainval_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    # Output file for convergence logging.
    convergence_name = 'convergence_split' + str(args.split_index) + '_run' + str(args.run) + '.csv'
    if not args.resume:
        with open(os.path.join(args.output, convergence_name), 'w') as fconv:
            fconv.write('epoch,split,metric,value\n')
    
    # Main training loop.
    if args.resume:
        epochs = range(ch['epoch']+1, args.nepochs+1)
    else:
        epochs = range(args.nepochs+1)
    
    for epoch in epochs:
        if args.optimizer == 'sgd':
            utils.adjust_learning_rate(optimizer, epoch, args.lr_anneal, args.lr)
        
        # Training step.
        if epoch > 0:
            loss = train(epoch, train_loader, model, criterion_class, criterion_loc, lambda_weight, optimizer)
        else:
            loss = np.nan
        with open(os.path.join(args.output, convergence_name), 'a') as fconv:
            fconv.write('{},train,loss,{}\n'.format(epoch, loss))
        
        # Validation: compute metrics and validation loss (evaluate primary classification only).
        train_probs = test(epoch, trainval_loader, model)
        train_auc, train_ber, train_fpr, train_fnr = train_dset.errors(train_probs)
        val_probs = test(epoch, val_loader, model)
        val_auc, val_ber, val_fpr, val_fnr = val_dset.errors(val_probs)
        val_loss = validate_loss(val_loader, model, criterion_class, criterion_loc, lambda_weight)
        
        print('Epoch: [{}/{}]\tLoss: {:.6f}\tTrain AUC: {:.4f}\tVal AUC: {:.4f}'.format(
            epoch, args.nepochs, loss, train_auc, val_auc))
        with open(os.path.join(args.output, convergence_name), 'a') as fconv:
            fconv.write('{},train,auc,{}\n'.format(epoch, train_auc))
            fconv.write('{},train,ber,{}\n'.format(epoch, train_ber))
            fconv.write('{},train,fpr,{}\n'.format(epoch, train_fpr))
            fconv.write('{},train,fnr,{}\n'.format(epoch, train_fnr))
            fconv.write('{},validation,loss,{}\n'.format(epoch, val_loss))
            fconv.write('{},validation,auc,{}\n'.format(epoch, val_auc))
            fconv.write('{},validation,ber,{}\n'.format(epoch, val_ber))
            fconv.write('{},validation,fpr,{}\n'.format(epoch, val_fpr))
            fconv.write('{},validation,fnr,{}\n'.format(epoch, val_fnr))
        
        # Save checkpoint.
        obj = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.lr_scheduler.state_dict() if lr_scheduler else None,
            'early_stopping': {
                'best_loss': early_stopping.best_loss,
                'counter': early_stopping.counter,
                'early_stop': early_stopping.early_stop,
                'min_delta': early_stopping.min_delta,
                'patience': early_stopping.patience
            } if early_stopping else None,
            'auc': val_auc,
        }
        torch.save(obj, os.path.join(args.output, 'checkpoint_split' +
                   str(args.split_index) + '_run' + str(args.run) + '.pth'))
        
        if args.lr_scheduler:
            lr_scheduler(-val_auc)
        if args.early_stopping:
            early_stopping(-val_auc)
            if early_stopping.early_stop:
                break

def test(epoch, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset)).cuda()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            # For evaluation, we consider only the primary classification output.
            output_class, _ = model(input)
            p = F.softmax(output_class, dim=1)
            probs[i*args.batch_size : i*args.batch_size + input.size(0)] = p.detach()[:, 1].clone()
    return probs.cpu().numpy()

def train(epoch, loader, model, criterion_class, criterion_loc, lambda_weight, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        # Expect target as a tuple: (target_class, target_loc)
        target_class, target_loc = target
        target_class_1hot = F.one_hot(target_class.long(), num_classes=2).cuda()
        target_loc = target_loc.long().cuda()
        output_class, output_loc = model(input)
        loss_class = criterion_class(output_class, target_class_1hot.float())
        loss_loc = criterion_loc(output_loc, target_loc)
        loss = loss_class + lambda_weight * loss_loc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * input.size(0)
    return running_loss / len(loader.dataset)

if __name__ == '__main__':
    main()
