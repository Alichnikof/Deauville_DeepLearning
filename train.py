import splitting.dataset as dataset
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

from torch.utils.data import  WeightedRandomSampler

parser = argparse.ArgumentParser(description='PET lymphoma classification')

# I/O PARAMS
parser.add_argument('--output', type=str, default='training_results',
                    help='name of output folder (default: "results")')

# MODEL PARAMS
parser.add_argument('--normalize', action='store_true', default=True,
                    help='normalize images')
parser.add_argument('--checkpoint', default='', type=str,
                    help='model checkpoint if any (default: none)')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume from checkpoint')

# New arguments for classifier architecture
parser.add_argument('--cls_arch', type=str, default='simple', choices=['simple', 'complex'],
                    help='Classifier architecture: "simple" (default) or "complex"')
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='Hidden dimension for complex classifier (if used)')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate for complex classifier (if used)')

# OPTIMIZATION PARAMS
parser.add_argument('--optimizer', default='sgd', type=str,
                    help='The optimizer to use (default: sgd)')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='learning rate (default: 1e-2)')
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
parser.add_argument('--augm', default=0, type=int, choices=[0, 1, 2, 3, 12, 4, 5, 14, 34, 45],
                    help='augmentation procedure 0=none,1=flip,2=rot,3=flip LR, 12=flip+rot, 4=scale, 5=noise, 14=flip+scale, 34=flipLR+scale, 45=scale+noise (default: 0)')
parser.add_argument('--balance', action='store_true', default=False,
                    help='balance dataset (balance loss)')
# New flag: --oversample to trigger oversampling with a WeightedRandomSampler
parser.add_argument('--oversample', action='store_true', default=False,
                    help='use oversampling with a WeightedRandomSampler')
parser.add_argument('--lr_scheduler', action='store_true', default=False,
                    help='decrease LR on plateau')
parser.add_argument('--early_stopping', action='store_true', default=False,
                    help='use early stopping')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='if set, only fine tune the classifier (fc) and the last block of the feature extractor')
parser.add_argument('--transfer_learning', action='store_true', default=False,
                    help='if set, freeze the feature extractor and only train the classifier head')



#Best hyperparameter from the study : 0.01 using SGD, ealy stopping, mini batch = 100
# 10 hours for 9 epoch...

def validate_loss(loader, model, criterion):
    """Compute average loss over the validation set."""
    model.eval()
    running_loss = 0.
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target_1hot = F.one_hot(target.long(), num_classes=2).cuda()
            y = model(input)
            loss = criterion(y, target_1hot.float())
            running_loss += loss.item() * input.size(0)
    return running_loss / len(loader.dataset)




def main():
    
    # Get user input
    global args
    args = parser.parse_args()
    print(args)
    best_auc = 0.

    # Output directory creation
    if not os.path.isdir(args.output):
        try:
            os.mkdir(args.output)
        except OSError:
            print('Creation of the output directory "{}" failed.'.format(args.output))
        else:
            print('Successfully created the output directory "{}".'.format(args.output))

    # Instantiate the model using the new classifier architecture options.
    model = utils.get_model(cls_arch=args.cls_arch, hidden_dim=args.hidden_dim, dropout=args.dropout)
    
    if args.checkpoint:
        ch = torch.load(args.checkpoint, weights_only=False)
        model_dict = model.state_dict()
        if 'state_dict' in ch:
            pretrained_dict = {k: v for k, v in ch['state_dict'].items() if k in model_dict}
        else:
            pretrained_dict = {k: v for k, v in ch.items() if k in model_dict}
        print('Loaded [{}/{}] keys from checkpoint'.format(len(pretrained_dict), len(model_dict)))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.resume:
        resume_path = os.path.join(args.output, 'checkpoint_split' + str(args.split_index) + '_run' + str(args.run) + '.pth')
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
        
    # Mutually exclusive mode: if transfer_learning is True, only unfreeze classifier head;
    # if finetune is True (and transfer_learning is False), unfreeze the classifier head and the last block of the feature extractor.
    if args.transfer_learning and args.finetune:
        raise ValueError("Please choose only one mode: either transfer_learning or finetune, not both.")

    if args.transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        print("Transfer learning mode enabled: only classifier head is trainable.")
    elif args.finetune:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        for param in model.features[7].parameters():
            param.requires_grad = True
        print("Fine-tuning mode enabled: classifier head and last block of feature extractor are trainable.")
        
    # Create optimizer only for parameters with requires_grad=True
    optimizer = utils.create_optimizer(
        list(filter(lambda p: p.requires_grad, model.parameters())),
        args.optimizer, args.lr, args.momentum, args.wd
    )

    # (Resume optimizer state if needed)
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

    # Set datasets
    train_dset, trainval_dset, val_dset, _, balance_weight_neg_pos = dataset.get_datasets_singleview(
        transform, args.normalize, args.balance, args.split_index)
    
    print('Datasets train:{}, val:{}'.format(
        len(train_dset.df), len(val_dset.df)))
    print(
        f"Weight of each class, no tumor: {balance_weight_neg_pos[0]}, tumor: {balance_weight_neg_pos[1]}")


    # Set loss criterion
    if args.balance:
        w = torch.Tensor(balance_weight_neg_pos)
        print('Balance loss with weights:', balance_weight_neg_pos)
        criterion = nn.BCEWithLogitsLoss(pos_weight=w).cuda()
    else:
        criterion = nn.BCEWithLogitsLoss().cuda()
    

    # Early stopping
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

    # Set loaders
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    trainval_loader = torch.utils.data.DataLoader(trainval_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)



    # -----------------------------
    # Create training DataLoader: Use oversampling if --oversample flag is True
    # -----------------------------
    if args.oversample:
        # Extract targets from training dataset
        targets = [train_dset[i][1] for i in range(len(train_dset))]
        class_counts = np.bincount(targets)
        print("Class counts:", class_counts)
        weights_per_class = 1.0 / class_counts
        sample_weights = [weights_per_class[t] for t in targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
        print("Using oversampling with WeightedRandomSampler for training.")
    else:
        train_loader =torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    trainval_loader = torch.utils.data.DataLoader(trainval_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


    # Set output file for convergence
    convergence_name = 'convergence_split' + str(args.split_index) + '_run' + str(args.run) + '.csv'
    if not args.resume:
        with open(os.path.join(args.output, convergence_name), 'w') as fconv:
            fconv.write('epoch,split,metric,value\n')

    # Main training loop
    if args.resume:
        epochs = range(ch['epoch']+1, args.nepochs+1)
    else:
        epochs = range(args.nepochs+1)

    for epoch in epochs:
        if args.optimizer == 'sgd':
            utils.adjust_learning_rate(optimizer, epoch, args.lr_anneal, args.lr)
        elif args.optimizer == 'adam':
            # Optionally adjust the learning rate for Adam (many times you may rely on Adam’s built-in behavior)
            # For now, we'll just pass.
            pass
        # ... continue training loop


        # Training
        if epoch > 0:
            loss = train(epoch, train_loader, model, criterion, optimizer)
        else:
            loss = np.nan
        # Log training loss
        with open(os.path.join(args.output, convergence_name), 'a') as fconv:
            fconv.write('{},train,loss,{}\n'.format(epoch, loss))

        # Validation: compute metrics via test() and also compute validation loss
        train_probs = test(epoch, trainval_loader, model)
        train_auc, train_ber, train_fpr, train_fnr = train_dset.errors(train_probs)
        val_probs = test(epoch, val_loader, model)
        val_auc, val_ber, val_fpr, val_fnr = val_dset.errors(val_probs)
        val_loss = validate_loss(val_loader, model, criterion)

        print('Epoch: [{}/{}]\tLoss: {:.6f}\tTrain AUC: {:.4f}\tVal AUC: {:.4f}'.format(epoch, args.nepochs, loss, train_auc, val_auc))

        # Log validation metrics including loss
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


        # Create checkpoint dictionary
        obj = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.lr_scheduler.state_dict() if lr_scheduler else None,
            'early_stopping': {'best_loss': early_stopping.best_loss, 'counter': early_stopping.counter,
                               'early_stop': early_stopping.early_stop, 'min_delta': early_stopping.min_delta,
                               'patience': early_stopping.patience} if early_stopping else None,

            'auc': val_auc,
        }
        # Save checkpoint
        torch.save(obj, os.path.join(args.output, 'checkpoint_split' +
                   str(args.split_index)+'_run'+str(args.run)+'.pth'))

        # Early stopping
        if args.lr_scheduler:
            lr_scheduler(-val_auc)
        if args.early_stopping:
            early_stopping(-val_auc)
            if early_stopping.early_stop:
                break


def test(epoch, loader, model):
    # Set model in test mode
    model.eval()
    # Initialize probability vector
    probs = torch.FloatTensor(len(loader.dataset)).cuda()
    # Loop through batches
    with torch.no_grad():
        for i, (input, _) in enumerate(loader):
            # Copy batch to GPU
            input = input.cuda()
            # Forward pass
            y = model(input)  # features, probabilities
            p = F.softmax(y, dim=1)
            # Clone output to output vector
            probs[i*args.batch_size:i*args.batch_size +
                  input.size(0)] = p.detach()[:, 1].clone()
    return probs.cpu().numpy()


def train(epoch, loader, model, criterion, optimizer):
    # Set model in training mode
    model.train()
    # Initialize loss
    running_loss = 0.
    # Loop through batches
    for i, (input, target) in enumerate(loader):
        # Copy to GPU
        input = input.cuda()
        target_1hot = F.one_hot(target.long(), num_classes=2).cuda()
        # Forward pass
        y = model(input)  # features, probabilities
        # Calculate loss
        loss = criterion(y, target_1hot.float())
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Store loss
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)


if __name__ == '__main__':
    main()

