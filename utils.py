import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np


def model_prediction(chpnt, cls_arch="simple", hidden_dim=256, dropout=0.3):
    model = my_model(cls_arch=cls_arch, hidden_dim=hidden_dim, dropout=dropout)
    ch = torch.load(chpnt, weights_only=False)
    if 'state_dict' in ch:
        model.load_state_dict(ch['state_dict'])
    else:
        model.load_state_dict(ch)
    return model

def get_model(cls_arch="simple", hidden_dim=256, dropout=0.3):
    model = my_model(cls_arch=cls_arch, hidden_dim=hidden_dim, dropout=dropout)
    model = model.cuda()
    return model

class my_model(nn.Module):
    def __init__(self, cls_arch="simple", hidden_dim=256, dropout=0.3):
        super(my_model, self).__init__()
        # Load a pretrained ResNet34 model
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Adapt the first convolutional layer to accept a single channel
        conv1 = model.conv1.weight.detach().clone().mean(dim=1, keepdim=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = conv1
        
        # Extract all layers except the final fully-connected layer
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.flat = True
        in_features = model.fc.in_features
        
        # Define the classifier head based on the chosen architecture
        if cls_arch == "simple":
            self.fc = nn.Linear(in_features, 2)
        elif cls_arch == "complex":
            self.fc = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2)
            )
        else:
            raise ValueError("Invalid classifier architecture. Choose 'simple' or 'complex'.")
    
    def forward(self, x):
        features = self.features(x)
        # The ResNet34 features output is of shape (batch, in_features, 1, 1)
        # so we can flatten directly:
        features = features.view(features.size(0), -1)
        out = self.fc(features)
        return out

    def forward_feat(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        out = self.fc(features)
        return features, out

    
class my_model_multitask(nn.Module):
        
    def __init__(self):
            
        super(my_model_multitask, self).__init__()
        # Load a ResNet34 model pretrained on ImageNet.
        model = models.resnet50(weights='DEFAULT')
        # Adjust the first conv layer to accept single-channel input.
        conv1 = model.conv1.weight.detach().clone().mean(dim=1, keepdim=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = conv1
        # Remove the original fully-connected layer.
        # We'll use the feature extractor (all layers except the final FC).
        self.features = nn.Sequential(*list(model.children())[:-1])
        in_features = model.fc.in_features
        
        # Original classifier head (for binary classification)
        self.fc = nn.Linear(in_features, 2)
        
        # Additional classifier head for lesion localization (51 classes)
        self.fc_loc = nn.Linear(in_features, 51)
        
        # Flag to indicate if we need to flatten the features (typically True)
        self.flat = True

    def forward(self, x):
        # Shared feature extraction
        features = self.features(x)
        if self.flat:
            features = features.view(x.size(0), -1)
        
        # Original task output
        output_class = self.fc(features)
        
        # Localization branch output
        # Option A: Allow joint training (both heads update the shared features)
        output_loc = self.fc_loc(features)
        # Option B: Detach the features to avoid influencing the shared backbone with localization gradients:
        # output_loc = self.fc_loc(features.detach())
        
        return output_class, output_loc
    
    def forward_feat(self, x):
        features = self.features(x)
        if self.flat:
            features = features.view(x.size(0), -1)
        output_class = self.fc(features)
        output_loc = self.fc_loc(features)
        return features, output_class, output_loc


# Optionally, add a helper function to instantiate the multitask model:
def get_model_multitask():
    model = my_model_multitask()
    model = model.cuda()
    return model
# Function that creates the optimizer

def create_optimizer(model_or_params, mode, lr, momentum, wd):
    # If the input has a 'parameters' method, use that; otherwise assume it's a list of parameters.
    if hasattr(model_or_params, 'parameters'):
        params = model_or_params.parameters()
    else:
        params = model_or_params

    if mode == 'sgd':
        optimizer = optim.SGD(params, lr,
                              momentum=momentum, dampening=0,
                              weight_decay=wd, nesterov=True)
    elif mode == 'adam':
        optimizer = optim.Adam(params, lr=lr,
                               weight_decay=wd)
    return optimizer

# Function to anneal learning rate


def adjust_learning_rate(optimizer, epoch, period, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every period epochs"""
    lr = start_lr * (0.1 ** (epoch // period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0.001):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(
                f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(self, optimizer, patience=1, min_lr=1e-5, factor=0.1, cooldown=1, threshold=0.001):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.cooldown = cooldown
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            cooldown=self.cooldown,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


if __name__ == '__main__':
    model = my_model().cuda()
    print(model)
    from torchinfo import summary
    summary(model, input_size=(12, 1, 224, 224))
