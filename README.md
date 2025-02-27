[![Static Badge](https://badgen.net/static/python/3.11/blue)](https://www.python.org/downloads/release/python-31111/)
[![Static Badge](https://badgen.net/static/pytorch/2.6.0/blue)](https://pytorch.org/get-started/locally/)
[![Static Badge](https://badgen.net/static/flake8/passed/green)](https://flake8.pycqa.org/en/latest/)


# Medical Image Processing using Kolmogorov-Arnold Networks

## Workspace
```
├── ckpts
├── data_lists
├── runs
└── MedicalKAN
    ├── configs
    │   └── *.yaml
    ├── data
    ├── metrics
    ├── nn
    ├── requirements
    ├── utils
    ├── .flake8
    ├── .gitignore
    ├── README.md
    └── train.py
```

## Quick-start
```
> cd ~
> git clone https://github.com/MaksimPenkin/MedicalKAN.git
> cd MedicalKAN
> export WORKBENCH=<...>
> python3 train.py --help
{'cuda': True, 'device_count': 1, 'device_current': 0, 'device_name': 'NVIDIA GeForce RTX 4080 Laptop GPU'}

Command-line arguments:
usage: train.py [-h]

Command-line arguments

options:
  -h, --help        show this help message and exit
  --use_gpu         gpu index to be used.
  --seed            manual seed to be used.
  --engine          engine specification.
  --epochs          how many times to iterate over the dataset (default: 1).
  --limit_batches   how much of the dataset to use (default: 1.0).
```

## Acknowledgements
- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/pdf/2404.19756): Liu, Ziming, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, and Max Tegmark. "Kan: Kolmogorov-arnold networks." arXiv preprint arXiv:2404.19756 (2024).
- [Convolutional Kolmogorov–Arnold Networks](https://arxiv.org/pdf/2406.13155v1): Bodner, Alexander Dylan, Antonio Santiago Tepsich, Jack Natan Spolski, and Santiago Pourteau. "Convolutional Kolmogorov-Arnold Networks." arXiv preprint arXiv:2406.13155 (2024).
- [Chebyshev Polynomial-Based Kolmogorov-Arnold Networks](https://arxiv.org/html/2405.07200v1): SS, Sidharth. "Chebyshev polynomial-based kolmogorov-arnold networks: An efficient architecture for nonlinear function approximation." arXiv preprint arXiv:2405.07200 (2024).
- [GitHub-pykan](https://github.com/KindXiaoming/pykan)
- [GitHub-Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs)
- [GitHub-Chebyshev-KANs](https://github.com/SynodicMonth/ChebyKAN)
- [GitHub-X-KANeRF](https://github.com/lif314/X-KANeRF)

## Citation
```python
@inproceedings{penkin2024kolmogorov,
  title={Kolmogorov-Arnold Networks as Deep Feature Extractors for MRI Reconstruction},
  author={Penkin, Maksim and Krylov, Andrey},
  booktitle={Proceedings of the 2024 9th International Conference on Biomedical Imaging, Signal Processing},
  pages={92--97},
  year={2024}
}
```
