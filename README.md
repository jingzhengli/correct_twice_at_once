## Correct Twice at Once: Learning to Correct Noisy Labels for Robust Deep Learning
Under review at conference MM 2022.

### Introduction
This is a PyTorch implementation of ["Correct Twice at Once: Learning to Correct Noisy Labels for Robust Deep Learning"]. 

Requirements:

* Python 3.7
* PyTorch 1.8.0
* torchvision 0.9.0

### Train:

- The code can be run on `cifar10`,`cifar100`, and `Clothing1M` datasets, where the datasets can be downloaded automatically. 
    ```bash
    sh run.sh
    ```

### Log:

- We provided a training log of the dataset `Clothing1M` which could be used to visualize the training process through `tensorboard` for reference. The log file can be found at: https://mega.nz/folder/58dFFahZ#cCR-HsLBlzbHQ6L7ztXCXQ.

- Place the log file in ``logs/``, and then execute the command.
    ```bash
    tensorboard --logdir=/logs/ --host= `host address`
    ```

***Note***: Our code will be further improved to make it cleaner.