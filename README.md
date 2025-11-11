[![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3129/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.5.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/previous-versions/#v250)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![arXiv](https://img.shields.io/badge/DOI-10.48550/arXiv.2509.13508-b31b1b.svg)](https://arxiv.org/abs/2509.13508)
[![arXiv](https://img.shields.io/badge/DOI-10.1134/S0361768825700057-b31b1b.svg)](https://link.springer.com/article/10.1134/S0361768825700057)
[![arXiv](https://img.shields.io/badge/DOI-10.1145/3707172.3707186-b31b1b.svg)](https://dl.acm.org/doi/10.1145/3707172.3707186)

# FunKAN: Medical Image Processing using Kolmogorov-Arnold Networks

## ⏳ Project Status: Finished
### Updates
- ✅ [2025/08/01] **Paper submitted!** FunKAN: Functional Kolmogorov-Arnold Network for Medical Image Enhancement and Segmentation
- ✅ [2025/05/12] **Paper published!** MBA-KAN: Adaptive Method for Selecting Basis Functions in Kolmogorov–Arnold Networks for Magnetic Resonance Image Enhancement, *Programming and Computer Software*
- ✅ [2025/01/17] **Conference paper published!** *9th International Conference on Biomedical Imaging, Signal Processing:* Kolmogorov-Arnold Networks as Deep Feature Extractors for MRI Reconstruction
- ✅ [2024/11/20] **Poster session!** *1st International School-Conference on Tensor Methods in Mathematics and Artificial Intelligence:* Fourier Neural Networks and Kolmogorov-Arnold Neural Networks for MRI Reconstruction
- ✅ [2024/09/19] **Conference paper published!** *34rd International Conference Graphicon 2024:* Scale Invariance property of Kolmogorov-Arnold Networks for MRI Gibbs Ringing Suppression

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
> export ALBUMENTATIONS_NO_TELEMETRY=1
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

| Paper                                                                                        | Code                                                                                |
|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/pdf/2404.19756)                          | [GitHub-pykan](https://github.com/KindXiaoming/pykan)                               |
| [Convolutional Kolmogorov–Arnold Networks](https://arxiv.org/pdf/2406.13155v1)               | [GitHub-Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs)   |
| [Chebyshev Polynomial-Based Kolmogorov-Arnold Networks](https://arxiv.org/html/2405.07200v1) | [GitHub-Chebyshev-KANs](https://github.com/SynodicMonth/ChebyKAN)                   |
|                                                                                              | [GitHub-X-KANeRF](https://github.com/lif314/X-KANeRF)                               |


## Cite this Project
If you use this project in your research or wish to refer to the baseline results, please use the following BibTeX entry.

```python
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

## Contributions
Contributions are welcome. Please raise issues as necessary. 

## References
[1] Liu, Ziming, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, and Max Tegmark. "KAN: Kolmogorov-Arnold Networks." arXiv preprint arXiv:2404.19756 (2024).

[2] Liu, Ziming, Pingchuan Ma, Yixuan Wang, Wojciech Matusik, and Max Tegmark. "KAN 2.0: Kolmogorov-Arnold Networks Meet Science." arXiv preprint arXiv:2408.10205 (2024).

[3] Bodner, Alexander Dylan, Antonio Santiago Tepsich, Jack Natan Spolski, and Santiago Pourteau. "Convolutional Kolmogorov-Arnold Networks." arXiv preprint arXiv:2406.13155 (2024).

[4] SS, Sidharth, Keerthana AR, and Anas KP. "Chebyshev Polynomial-based Kolmogorov-Arnold Networks: An Efficient Architecture for Nonlinear Function Approximation." arXiv preprint arXiv:2405.07200 (2024).

[5] Seydi, Seyd Teymoor. "Exploring the Potential of Polynomial Basis Functions in Kolmogorov-Arnold Networks: A Comparative Study of Different Groups of Polynomials." arXiv preprint arXiv:2406.02583 (2024).

[6] Drokin, Ivan. "Kolmogorov-Arnold Convolutions: Design Principles and Empirical Studies." arXiv preprint arXiv:2407.01092 (2024).
