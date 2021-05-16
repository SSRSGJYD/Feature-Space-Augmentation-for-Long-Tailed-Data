# Feature-Space-Augmentation-for-Long-Tailed-Data

Unofficial implementation for ECCV 2020 paper "Feature Space Augmentation for Long-Tailed Data"



## Code Structure

    ./
    ├── configs/                            # store experiment configs
    │   ├── phase_i/                        # there are 3 training phase in the paper
    │   │   ├── cifar10-LT_resnet18.yaml    # a training config
    │   │   └── ... 
    │   ├── phase_ii/
    │   │   ├── cifar10-LT_resnet18.yaml    # config for extracting feature map
    │   │   └── ... 
    │   └── phase_iii/
    │   │   ├── cifar10-LT_resnet18.yaml    # a training config
    │   │   └── ... 
    ├── datasets/                           # all datasets
    │   ├── __init__.py                     # include get_dataset()
    │   ├── feature_dataset.py				# training dataset for phase 3
    │   ├── cifar_lt.py
    │   └── ...
    ├── models/                             # all models
    │   ├── __init__.py                     # include get_model()
    │   ├── resnet.py
    │   └── ...
    ├── utils/                              # Tools and utilities
    ├── checkpoints/                        # save model checkpoints
    │   ├── phase_i/
    │   │   ├── cifar10-LT_resnet18/        # same as current config filename
    │   │   │   ├── note_1/                 # pass to argparser to identify different experiment setting
    │   │   │   │   ├── best_model.pt       # checkpoint with best test acc
    │   │   │   │   ├── model_epoch_0020.pt # checkpoint at epoch 20
    │   │   │   │   └── ... 
    │   │   │   ├── note_2/
    │   │   │   └── ...
    │   │   └── ...
    │   └── phase_iii/
    ├── log/                                # logging files and backup configs, same structure as 
    ├── phase_i_train_or_test.py			# phase 1 training & test script
    ├── phase_ii_extract_feature.py			# phase 2 script
    ├── phase_iii_train_or_test.py			# phase 3 training & test script
    ├── run_summary/                        # tensorboard summary, same structure as ./checkpoints
    ├── .gitignore                          
    ├── LICENSE
    ├── requirements.txt                   
    └── README.md

