# Adaptive Split Federated Learning (ASFL)

A PyTorch framework for **split federated learning** with **personalisation** and **out-of-distribution (OOD) aware inference**. Each client runs the front part of a network locally, the server trains the shared back part, and clients can then personalise their models while the framework decides — per sample or per batch — whether to use the generalised or the personalised model.

The framework uses a **key-value store** of front-model activations so the server can train without repeatedly re-running client forward passes, which makes split training cheaper in compute and communication.

## Highlights

- **Split learning** between clients and a central server, with configurable split points (`--split`).
- **Three training variants**
  - **ESL** — Efficient Split Learning with the standard cross-entropy objective.
  - **RSL** — training with a combined loss (cross-entropy + supervised contrastive loss).
  - **RSL-E** — the combined loss with the contrastive term computed on the server side.
- **Individual personalisation (`_IP`)** — each client fine-tunes its own copy after the shared training stage.
- **Hybrid inference** — samples flagged as OOD for a client are routed to the generalised model, the rest to the personalised one (sample-wise and batch-wise modes).
- **Non-IID data** — client splits are drawn from a Dirichlet distribution (α = 0.3), with optional OOD clients (`--ood`).
- Experiment tracking with **Weights & Biases**.

## Supported datasets

| Dataset | Domain | Default model |
|---|---|---|
| CIFAR-10 | Natural images | ResNet-18 |
| Fashion-MNIST | Clothing images | ResNet-18 |
| ISIC 2019 | Dermoscopy (skin lesions) | ResNet-18 |
| Office-Home | Multi-domain images | ResNet-50 |
| PACS | Multi-domain images | ResNet-50 |
| VLCS | Multi-domain images | ResNet-50 |

## Repository structure

```
.
├── trainer.py                  # Main entry point — picks a trainer from --dataset
├── threshold_finder.py         # Calibrates OOD thresholds and runs hybrid inference
├── config.py                   # Global settings (e.g. W&B key)
├── *_esl_script.sh             # ESL experiment scripts, one per dataset
├── *_rsl_script.sh             # RSL experiment scripts, one per dataset
└── ImageClassification_Task/
    ├── ic_client*.py           # Client-side (front model) logic
    ├── ic_server*.py           # Server-side (back model) logic
    ├── ic_trainer*.py          # CIFAR-10 trainers for each variant
    ├── ic_fmnist_trainer*.py   # Fashion-MNIST trainers
    ├── *builder.py             # Dataset loading and Dirichlet client splits
    ├── combined_loss_fn.py     # Cross-entropy + supervised contrastive loss
    ├── contrastive_loss_fn.py  # Supervised contrastive (SupCon) loss
    ├── focal_loss_fn.py        # Focal loss
    ├── ISIC2019/               # ISIC 2019 builders, clients and trainers
    ├── officehome/             # Office-Home builders, clients and trainers
    ├── pacs/                   # PACS builders, clients and trainers
    └── vlcs/                   # VLCS builders, clients and trainers
```

## Installation

```bash
git clone https://github.com/Manisha-IITBh/ASFL.git
cd ASFL
pip install torch torchvision torchmetrics numpy pandas scikit-learn scipy matplotlib seaborn tqdm wandb requests
```

To log to Weights & Biases, set your key in `config.py` (`WANDB_KEY = "..."`) or run `wandb login`.

## Data

- **CIFAR-10** and **Fashion-MNIST** download automatically to `./data`.
- **ISIC 2019**, **Office-Home**, **PACS** and **VLCS** must be downloaded separately. Update the dataset paths in the matching `*builder.py` file before running (for example, `ImageClassification_Task/ISIC2019/ISICbuilder.py`).

## Usage

Run a single experiment:

```bash
python trainer.py -c 10 -bs 64 -tbs 256 -n 60 \
  --client_lr 1e-3 --server_lr 1e-3 \
  --dataset CIFAR10_ESL_IP --model resnet18 --split 1 \
  -kv --kv_refresh_rate 0 --kv_factor 1 \
  --ood 0 --seed 42 --exp_name cifar_esl_run --wandb
```

Or run a full set of experiments for a dataset:

```bash
bash cifar_esl_script.sh
bash pacs_rsl_script.sh
```

Logs are written to a per-dataset output folder (e.g. `./CIFAR/ESL/`), and checkpoints to `./saved_models/<dataset>/...`.

### Choosing a dataset and variant

`--dataset` takes the form `<DATASET>_<VARIANT>`, for example:

| Variant | CIFAR-10 | Fashion-MNIST | ISIC 2019 | Office-Home / PACS / VLCS |
|---|---|---|---|---|
| ESL + personalisation | `CIFAR10_ESL_IP` | `FMNIST_ESL_IP` | `ISIC2019_ESL_IP` | `PACS_ESL_IP`, … |
| RSL + personalisation | `CIFAR10_CL_IP` | `FMNIST_CL_IP` | `ISIC2019_RSL_IP` | `PACS_RSL_IP`, … |
| RSL-E + personalisation | — | `FMNIST_RSLE_IP` | `ISIC2019_RSLE_IP` | `PACS_RSLE_IP`, `VLCS_RSLE_IP` |

The trainer imports in `trainer.py` are grouped by dataset. Uncomment the group for the dataset you want to run.

### Main arguments

| Argument | Description |
|---|---|
| `-c` | Number of clients |
| `-bs` / `-tbs` | Training / test batch size |
| `-n` | Number of epochs |
| `--client_lr`, `--server_lr` | Learning rates for the client and server models |
| `--model` | Backbone (`resnet18`, `resnet50`) |
| `--split` | Where the network is split between client and server |
| `-kv` | Use the key-value activation store |
| `--kv_factor`, `--kv_refresh_rate` | How many times the store is populated, and how often it is refreshed |
| `--ood` | OOD setting for the client data split (`0` = none) |
| `--seed` | Random seed |
| `--exp_name` | Experiment name used for logs and checkpoints |
| `--wandb` | Enable Weights & Biases logging |

### OOD threshold calibration and hybrid inference

After training, run:

```bash
python threshold_finder.py
```

This loads the best saved models, calibrates per-client OOD thresholds, and reports **generalised**, **personalised**, and **hybrid** (sample-wise and batch-wise) inference results.

## License

Released under the [MIT License](LICENSE).
