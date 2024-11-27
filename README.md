[![Static Badge](https://badgen.net/static/python/3.9/blue)](https://www.python.org/downloads/release/python-3913/)
[![Static Badge](https://badgen.net/static/pytorch/2.4.0/blue)](https://pytorch.org/get-started/locally/)
[![Static Badge](https://badgen.net/static/flake8/passed/green)](https://flake8.pycqa.org/en/latest/)


# Medical Image Processing using Kolmogorov-Arnold Networks

## Workspace
```
├── ckpts
├── logs
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
    ├── train.py
    └── test.py
```

## Quick-start
```
> cd ~
> git clone https://github.com/MaksimPenkin/MedicalKAN.git
> cd MedicalKAN
> python3 train.py --help
```

## Acknowledgements
- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/pdf/2404.19756): Liu, Ziming, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, and Max Tegmark. "Kan: Kolmogorov-arnold networks." arXiv preprint arXiv:2404.19756 (2024).
- [Convolutional Kolmogorov–Arnold Networks](https://arxiv.org/pdf/2406.13155v1): Bodner, Alexander Dylan, Antonio Santiago Tepsich, Jack Natan Spolski, and Santiago Pourteau. "Convolutional Kolmogorov-Arnold Networks." arXiv preprint arXiv:2406.13155 (2024).
- [Chebyshev Polynomial-Based Kolmogorov-Arnold Networks](https://arxiv.org/html/2405.07200v1): SS, Sidharth. "Chebyshev polynomial-based kolmogorov-arnold networks: An efficient architecture for nonlinear function approximation." arXiv preprint arXiv:2405.07200 (2024).
- [GitHub-pykan](https://github.com/KindXiaoming/pykan)
- [GitHub-Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs)
- [GitHub-Chebyshev-KANs](https://github.com/SynodicMonth/ChebyKAN)
