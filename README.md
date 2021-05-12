# Feature-Space-Augmentation-for-Long-Tailed-Data

***
Unofficial implementation for ECCV 2020 paper "Feature Space Augmentation for Long-Tailed Data"

## Code Structure

    .
    ├── configs                             # store experiment configs
    │   ├── phase_i                         # there are 3 training phase in the paper
    │   │   ├── cifar10-LT_resnet18.yaml    # a training config
    │   │   └── ... 
    │   ├── phase_ii
    │   └── phase_iii
    ├── datasets                            # all datasets
    │   ├── __init__.py                     # include get_dataset()
    │   ├── cifar_lt.py
    │   └── ...
    ├── models                              # all models
    │   ├── __init__.py                     # include get_model()
    │   ├── resnet.py
    │   └── ...
    ├── process                             # train, test code for 3 phases
    │   ├── phase_i_train_or_test.py
    │   └── ...
    ├── utils                               # Tools and utilities
    ├── checkpoints                         # save model checkpoints
    │   ├── phase_i
    │   │   ├── cifar10-LT_resnet18         # same as current config filename
    │   │   │   ├── note_1                  # pass to argparser to identify different experiment setting
    │   │   │   ├── note_2
    │   │   │   └── ...
    │   │   └── ...
    │   ├── phase_ii
    │   └── phase_iii
    ├── log                                 # logging files and backup configs, same structure as ./checkpoints
    ├── run_summary                         # tensorboard summary, same structure as ./checkpoints
    ├── .gitignore                          
    ├── LICENSE
    ├── requirements.txt                   
    └── README.md
