import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import optuna
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
import numpy as np

import utils          # contains your model definitions (e.g. my_model)
import splitting.dataset as dataset  # contains your dataset classes and get_datasets_singleview function

# --------------------------
# Define augmentation choices and a helper function.
# --------------------------
AUG_CHOICES = [0, 1, 2, 3, 12, 4, 5, 14, 34, 45]

def get_transform(aug_choice):
    flipHorVer = dataset.RandomFlip()
    flipLR = dataset.RandomFlipLeftRight()
    rot90 = dataset.RandomRot90()
    scale = dataset.RandomScale()
    noise = dataset.RandomNoise()
    
    if aug_choice == 0:
        return None
    elif aug_choice == 1:
        return transforms.Compose([flipHorVer])
    elif aug_choice == 2:
        return transforms.Compose([rot90])
    elif aug_choice == 3:
        return transforms.Compose([flipLR])
    elif aug_choice == 12:
        return transforms.Compose([flipHorVer, rot90])
    elif aug_choice == 4:
        return transforms.Compose([scale])
    elif aug_choice == 5:
        return transforms.Compose([noise])
    elif aug_choice == 14:
        return transforms.Compose([flipHorVer, scale])
    elif aug_choice == 34:
        return transforms.Compose([flipLR, scale])
    elif aug_choice == 45:
        return transforms.Compose([scale, noise])
    else:
        return None

# --------------------------
# Define the model architecture with optional modification to the classification head.
# --------------------------
def define_model(trial):
    """
    Create the model using your existing architecture, with an optional modification
    to the classification layer.
    """
    model = utils.my_model()
    
    # Choose the architecture for the classification head.
    # "simple": use the current single linear layer.
    # "complex": add an extra hidden layer with dropout.
    cls_arch = trial.suggest_categorical("cls_arch", ["simple", "complex"])
    
    if cls_arch == "complex":
        in_features = model.fc.in_features  # original input features for classifier
        hidden_dim = trial.suggest_int("hidden_dim", 64, 512, step=64)
        dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)
        
        # Replace the original fc layer with a sequential module.
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)  # output logits for two classes
        )
    # else: keep the simple linear classifier (the default in utils.my_model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model

# --------------------------
# Define the Optuna objective function.
# --------------------------
def objective(trial, split_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Suggest hyperparameters.
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["sgd", "adam"])
    wd = trial.suggest_float("wd", 1e-5, 1e-2, log=True)
    use_lr_scheduler = trial.suggest_categorical("lr_scheduler", [True, False])
    use_balance = trial.suggest_categorical("balance", [True, False])
    oversample = trial.suggest_categorical("oversample", [True, False])
    # New hyperparameter: augmentation.
    augm_choice = trial.suggest_categorical("augm", AUG_CHOICES)
    transform = get_transform(augm_choice)

    # Define the model (with possible changes in the classifier head).
    model = define_model(trial)
    
    # Define optimizer.
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
    
    # Optionally define a learning rate scheduler.
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=3, min_lr=1e-6, verbose=False
        )
    else:
        scheduler = None

    # --------------------------
    # Load datasets.
    # --------------------------
    # Pass the selected transform to your dataset loader.
    train_dset, trainval_dset, val_dset, _, balance_weight = dataset.get_datasets_singleview(
        transform, norm=True, balance=use_balance, split_index=split_index
    )
    
    # Use oversampling if selected.
    if oversample:
        targets = [train_dset[i][1] for i in range(len(train_dset))]
        class_counts = np.bincount(targets)
        weights_per_class = 1.0 / class_counts
        sample_weights = [weights_per_class[t] for t in targets]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=batch_size, sampler=sampler, num_workers=4
        )
        print("Using oversampling with WeightedRandomSampler for training.")
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=batch_size, shuffle=True, num_workers=4
        )

    val_loader = torch.utils.data.DataLoader(
        val_dset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    # --------------------------
    # Define loss function.
    # --------------------------
    if use_balance:
        pos_weight = torch.tensor(balance_weight[1] / balance_weight[0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    
    # --------------------------
    # Training loop.
    # --------------------------
    num_epochs = 10  # Optimization runs for 10 epochs per trial.
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
        
        # --------------------------
        # Validation loop.
        # --------------------------
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
        
        print(f"Split {split_index} Epoch {epoch}: Train Loss={train_loss:.4f}, Val AUC={val_auc:.4f}")
        
        if scheduler is not None:
            scheduler.step(val_auc)
        
        trial.report(val_auc, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
    
    return best_val_auc

# --------------------------
# Run the optimization for each split.
# --------------------------
if __name__ == "__main__":
    for split_index in range(0, 20):
        print(f"\n=== Running optimization for split {split_index} ===")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, split_index), n_trials=20, timeout=3600)
        
        best_trial = study.best_trial
        print(f"\nBest trial for split {split_index}:")
        print(f"  Value (best Val AUC): {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
            
        df_results = study.trials_dataframe()
        df_results.to_csv(f"optuna_results_split{split_index}.csv", index=False)
