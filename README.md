
# Binary Deauville Classification 2025

This repository accompanies the master's thesis project on automated PET image classification for lymphoma.

We present a simplified and modular framework for binary Deauville scoring (DS 1–3 vs. 4–5) from coronal and sagittal PET MIP views. The architecture builds on ResNet34 and reproduces the core functionality of the LARS framework described in Häggström et al. (2023), with new features for multitask learning, fine-tuning, and automated hyperparameter optimization.

## 🔧 Framework

* Python 3.10+
* PyTorch

---

## 📁 Repository Structure

* `train.py` — Train a 2D ResNet34 classifier for binary Deauville scoring.
* `train_loca.py` — Train a multitask ResNet model predicting Deauville score and lesion localization (14 anatomical classes).
* `predict.py` — Inference using a binary classifier.
* `predict_loca.py` — Inference using a multitask model.
* `find_best_model.py` — Select best-performing runs (per split) based on AUC.
* `optuna_finetuning.py` — Fine-tuning from pretrained checkpoints with Optuna.
* `optun.py` — Automated hyperparameter search using Optuna.
* `utils.py` — Model definitions, optimizers, schedulers, early stopping.
* `dataset.py`, `dataset_loca.py` — Dataset definitions for both tasks.
* `prepro_script.py` — Utilities for SUV computation and MIP generation from DICOM.
* `data.csv`, `data_resplit.csv` — CSV files with paths, labels, splits, and matrix sizes.

---

## 📄 Input Format

Each scan includes two views: coronal and sagittal PET MIPs. The dataset CSV must follow this format:

```
scan_id,filename,target,matrix_size_1,matrix_size_2,split0,split1,...
0,image_0_cor.bin,1,250,250,train,...
0,image_0_sag.bin,1,180,250,train,...
...
```

The Deauville score is binarized: 0 = DS1–3 (good prognosis), 1 = DS4–5 (poor prognosis).

---

## 🚀 How to Run

1. **Prepare your CSV and MIP data.**
2. **Train models across splits:**

```bash
python train.py --split_index 0 --run 1 --output results --cls_arch complex --optimizer sgd --lr 0.01 --early_stopping
```

3. **Find best runs per split:**

```bash
python find_best_model.py --dir results
```

4. **Make predictions:**

```bash
python predict.py --output results --chptfolder results --splits 0 19
```

5. **Aggregate (LARS-avg or LARS-max):**

   * Average across top-10 predictions per scan and view.

---

## 🧪 Multitask Extension

Use `train_loca.py` and `predict_loca.py` to jointly predict:

* Binary Deauville class
* Lesion localization (14-class anatomical site prediction)

Control the loss balance via a lambda coefficient (`λ × localization_loss`).

---

## 🔍 Hyperparameter Optimization

Use Optuna to explore model architectures and training strategies:

```bash
python optun.py
python optuna_finetuning.py
```

Fine-tuning modes include:

* Transfer learning
* Fine-tuning last residual block + head
* Full retraining

---

## 🧠 Model Architecture

* Based on **ResNet34** pretrained on ImageNet
* Accepts **1-channel input** (PET MIP)
* Classifier head: `simple` (linear) or `complex` (MLP with dropout)

For multitask:

* Second head for 14-class localization

---

## 📥 Pretrained Models

Top-10 models available upon request

---

## 📜 License

This work is shared under the **Creative Commons Attribution-NonCommercial 4.0 International License**. See `LICENSE-CC-BY-NC-4.0.md`.

---

## 📚 Reference

```
@article{haggstrom2023lymphoma,
  title={Deep learning for [18F]fluorodeoxyglucose-PET-CT classification in patients with lymphoma: a dual-centre retrospective analysis},
  author={H\u00e4ggstr\u00f6m, Ida and Leithner, Doris and Alv\u00e9n, Jennifer and others},
  journal={The Lancet Digital Health},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/S2589-7500(23)00203-0}
}
```

---

## 📬 Contact

Ahmad Mezher — ahmadalimezher@gmail.com
Master’s Thesis @ ULB — 2025
