[![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3129/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.5.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/previous-versions/#v250)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


# FUNKAN: Medical Image Processing using Kolmogorov-Arnold Networks

![alt text](./example.png)

## 📌 Workspace
```
└── $WORKBENCH
    ├── ckpts
    ├── data_lists
    ├── runs
    └── MedicalKAN
        ├── configs
        │   └── *.yaml
        ├── requirements
        ├── src
        │   ├── data
        │   ├── models
        │   ├── nn
        │   ├── trainers
        │   └── utils
        ├── .flake8
        ├── .gitignore
        ├── example.png
        ├── README.md
        ├── ruff.toml
        └── train.py
```

## 📌 Installation
You may find requirements
[here](https://github.com/MaksimPenkin/MedicalKAN/tree/main/requirements).

## 📌 Quick-start
```
> cd ~
> git clone https://github.com/MaksimPenkin/MedicalKAN.git
> cd MedicalKAN
> export WORKBENCH=<...>
> export DATASETS=<...>
> python3 train.py --help
usage: train.py [-h]

options:
  -h, --help            show this help message and exit
  --config              path to experiment configuration file (*.yaml).
  --limit_train_batches 
                        how much of training dataset to use (default: 1.0).
  --limit_val_batches   how much of validation dataset to use (default: 1.0).
  --seed                if specified, sets the seed for pseudo-random number generators.
```

## Acknowledgements
Big thank you for the awesome works!

| arXiv                                                                                        | GitHub                                                                            |
|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/pdf/2404.19756)                          | [GitHub-pykan](https://github.com/KindXiaoming/pykan)                             |
| [Convolutional Kolmogorov–Arnold Networks](https://arxiv.org/pdf/2406.13155v1)               | [GitHub-Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs) |
| [Chebyshev Polynomial-Based Kolmogorov-Arnold Networks](https://arxiv.org/html/2405.07200v1) | [GitHub-Chebyshev-KANs](https://github.com/SynodicMonth/ChebyKAN)                 |
|                                                                                              | [GitHub-X-KANeRF](https://github.com/lif314/X-KANeRF)                                                                        |


## Citation

```python
@abbr.: FUNKAN / ex-MBA-KAN

@inproceedings{penkin2025kolmogorov,
  title={Adaptive Method for Selecting Basis Functions in Kolmogorov–Arnold Networks for Magnetic Resonance Image Enhancement},
  author={Penkin, Maksim and Krylov, Andrey},
  booktitle={Programming and Computer Software},
  pages={167--172},
  year={2025}
}
```

```python
@inproceedings{penkin2024kolmogorov,
  title={Kolmogorov-Arnold Networks as Deep Feature Extractors for MRI Reconstruction},
  author={Penkin, Maksim and Krylov, Andrey},
  booktitle={Proceedings of the 2024 9th International Conference on Biomedical Imaging, Signal Processing},
  pages={92--97},
  year={2024}
}
```
