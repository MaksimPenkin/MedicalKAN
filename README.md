[![Static Badge](https://badgen.net/static/python/3.9/blue)](https://www.python.org/downloads/release/python-3913/)
[![Static Badge](https://badgen.net/static/pytorch/2.4.0/blue)](https://pytorch.org/get-started/locally/)
[![Static Badge](https://badgen.net/static/flake8/passed/green)](https://flake8.pycqa.org/en/latest/)


# Medical Image Processing using Kolmogorov-Arnold Networks

## Workspace
```

├── ckpts
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
    ├── train.py
    └── test.py
```

## Quick-start

```
> cd ~
> git clone https://github.com/MaksimPenkin/MedicalKAN.git
> cd MedicalKAN
> python train.py --help
```

## Acknowledgements
- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/pdf/2404.19756)
- [Convolutional Kolmogorov–Arnold Networks](https://arxiv.org/pdf/2406.13155v1)
- [GitHub-pykan](https://github.com/KindXiaoming/pykan)
- [GitHub-Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs)
