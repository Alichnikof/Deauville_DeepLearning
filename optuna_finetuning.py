import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import optuna
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

import utils          # contains your model definitions (e.g. my_model, get_model)
import splitting.dataset as dataset  # contains your dataset classes and get_datasets_singleview function

# --------------------------
# Fine-tuning model initialization
# --------------------------
def fine_tuning_define_model(trial):
    """
    Loads the model and initializes it from a checkpoint from the article.
    The checkpoint is assumed to be at a fixed path (adjust as needed).
    """
    # Get model (e.g., ResNet34-based model) from utils
    model = utils.get_model(cls_arch="simple", hidden_dim=256, dropout=0.3)
    
    # Specify the checkpoint path from the article (adjust as needed)
    checkpoint_path = "/home/mezher/Documents/Deauville_DeepLearning/checkpoints/checkpoint_split0_run0.pth"
    if os.path.exists(checkpoint_path):
        ch = torch.load(checkpoint_path, map_location="cpu")
        model_dict = model.state_dict()
        if 'state_dict' in ch:
            pretrained_dict = {k: v for k, v in ch['state_dict'].items() if k in model_dict}
        else:
            pretrained_dict = {k: v for k, v in ch.items() if k in model_dict}
        print(f"Loaded [{len(pretrained_dict)}/{len(model_dict)}] keys from checkpoint.")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print("Checkpoint not found:", checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

# --------------------------
# Define the Optuna objective function for fine tuning.
# --------------------------
def objective_finetune(trial, split_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Suggest hyperparameters for fine tuning:
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["sgd", "adam"])
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    use_lr_scheduler = trial.suggest_categorical("lr_scheduler", [True, False])
    use_balance = trial.suggest_categorical("balance", [True, False])
    # Now include a third option "full_retrain" so that we can retrain the full model.
    ft_mode = trial.suggest_categorical("ft_mode", ["transfer_learning", "finetune", "full_retrain"])
    oversample_flag = trial.suggest_categorical("oversample", [True, False])
    
    # Create model by loading the pretrained checkpoint.
    model = fine_tuning_define_model(trial)
    
    # Set fine-tuning mode based on trial parameter:
    if ft_mode == "transfer_learning":
        # Freeze entire feature extractor; train only classifier head.
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        print("Using transfer learning: only classifier head is trainable.")
    elif ft_mode == "finetune":
        # Freeze all layers, then unfreeze classifier head and the last block.
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        # Assuming model.features[7] corresponds to the last block.
        for param in model.features[7].parameters():
            param.requires_grad = True
        print("Using fine tuning: classifier head and last block are trainable.")
    elif ft_mode == "full_retrain":
        # Do nothing, so all parameters remain trainable.
        print("Using full retraining: all model parameters are trainable.")
    
    # Define optimizer using the suggested parameters.
    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, momentum=0.9, weight_decay=wd, nesterov=True
        )
    else:  # adam
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=wd
        )
    
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=3, min_lr=1e-6, verbose=False
        )
    else:
        scheduler = None

    # --------------------------
    # Load datasets for fine tuning.
    # --------------------------
    # get_datasets_singleview returns (train_dset, trainval_dset, val_dset, test_dset, balance_weight)
    train_dset, trainval_dset, val_dset, _, balance_weight = dataset.get_datasets_singleview(
        transform=None, norm=True, balance=use_balance, split_index=split_index
    )
    
    # Create training DataLoader. If oversample flag is set, use WeightedRandomSampler.
    if oversample_flag:
        targets = [train_dset[i][1] for i in range(len(train_dset))]
        class_counts = np.bincount(targets)
        print("Class counts:", class_counts)
        weights_per_class = 1.0 / class_counts
        sample_weights = [weights_per_class[t] for t in targets]
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, 
                                                         num_samples=len(sample_weights), 
                                                         replacement=True)
        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, sampler=sampler, num_workers=4)
        print("Oversampling enabled.")
    else:
        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Define loss function. If use_balance is True, use balanced loss.
    if use_balance:
        pos_weight = torch.tensor(balance_weight[1] / balance_weight[0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    
    # --------------------------
    # Training loop for fine tuning.
    # --------------------------
    num_epochs = 12 # You can adjust the number of fine tuning epochs.
    best_val_auc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            targets_onehot = F.one_hot(targets.long(), num_classes=2).float()
            loss = criterion(outputs, targets_onehot)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except Exception:
            val_auc = 0.0
        
        print(f"FineTune Split {split_index} Epoch {epoch}: Train Loss={train_loss:.4f}, Val AUC={val_auc:.4f}")
        
        if scheduler is not None:
            scheduler.step(val_auc)
        
        trial.report(val_auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
    return best_val_auc

# --------------------------
# Run the fine tuning optimization for multiple splits.
# --------------------------
if __name__ == "__main__":
    best_runs = []

    #ONLE TO 9 cause no checkpoint over 9
    for split_index in range(0, 10):
        print(f"\n=== Fine tuning optimization for split {split_index} ===")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective_finetune(trial, split_index), n_trials=10, timeout=3600)
        
        best_trial = study.best_trial
        print(f"\nBest trial for fine tuning split {split_index}:")
        print(f"  Best Val AUC: {best_trial.value}")
        print("  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        
                    # Save the trial results to a CSV file
        df_results = study.trials_dataframe()
        df_results.to_csv(f"optuna_resultsfinetuning_split{split_index}.csv", index=False)

    