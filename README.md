# Conformal-Correction
Conformal correction on CIFAR-10/100 with PyTorch.

## Features
* Support for multiple GPUs  
* Training progress bar with comprehensive details  
* Unified interface for different network architectures

## Install
* Install [PyTorch](http://pytorch.org/)
* Clone recursively
  ```
  git clone --recursive https://github.com/Anonymity23143/Conformal-Correction.git
  ```

## Pre-Training
* Training commands
  ```
  # cifar10
  python cifar10_train.py -a resnet --depth 56 --epochs 160 --schedule 80 120 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-56

  python cifar10_train.py -a preresnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/preresnet-110 

  python cifar10_train.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint   checkpoints/cifar10/densenet-bc-100-12


  # cifar100
  python cifar100_train.py -a resnet --dataset cifar100 --depth 56 --epochs 160 --schedule 80 120 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-56 

  python cifar100_train.py -a preresnet --dataset cifar100 --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/preresnet-110

  python cifar100_train.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar100/densenet-bc-100-12
  ```

## Conformal correction
* Correction commands
  ```
  # cifar10-resnet
  python cifar10.py -a resnet --depth 56 --epochs 200 --schedule 80 120 --gamma 0.1 --wd 1e-4 -e True --resume checkpoints/cifar10/resnet-56 --correction conf

  python cifar10.py -a resnet --depth 56 --epochs 200 --schedule 80 120 --gamma 0.1 --wd 1e-4 -e True --resume checkpoints/cifar10/resnet-56 --correction focal4


  # cifar10-preresnet
  python cifar10.py -a preresnet --depth 110 --epochs 200 --schedule 80 120 --gamma 0.1 --wd 1e-4 -e True --resume checkpoints/cifar10/preresnet-110 --correction conf

  python cifar10.py -a preresnet --depth 110 --epochs 200 --schedule 80 120 --gamma 0.1 --wd 1e-4 -e True --resume checkpoints/cifar10/preresnet-110 --correction focal4


  # cifar10-densenet
  python cifar10.py -a densenet --depth 100 --epochs 200 --schedule 80 120 --gamma 0.1 --wd 1e-4 -e True --resume checkpoints/cifar10/densenet-bc-100-12 --correction conf

  python cifar10.py -a densenet --depth 100 --epochs 200 --schedule 80 120 --gamma 0.1 --wd 1e-4 -e True --resume checkpoints/cifar10/densenet-bc-100-12 --correction focal4


  # cifar100-resnet
  python cifar100.py -a resnet --dataset cifar100 --depth 56 --epochs 500 --schedule 80 120 --gamma 0.1 --wd 1e-4 -e True --resume checkpoints/cifar100/resnet-56 --correction conf

  python cifar100.py -a resnet --dataset cifar100 --depth 56 --epochs 500 --schedule 80 120 --gamma 0.1 --wd 1e-4 -e True --resume checkpoints/cifar100/resnet-56 --correction focal4


  # cifar100-preresnet
  python cifar100.py -a preresnet --dataset cifar100 --depth 110 --epochs 500 --schedule 80 120 --gamma 0.1 --wd 1e-4 -e True --resume checkpoints/cifar100/preresnet-110 --correction conf

  python cifar100.py -a preresnet --dataset cifar100 --depth 110 --epochs 500 --schedule 80 120 --gamma 0.1 --wd 1e-4 -e True --resume checkpoints/cifar100/preresnet-110 --correction focal4


  # cifar100-densenet
  python cifar100.py -a densenet --dataset cifar100 --depth 100 --epochs 500 --schedule 80 120 --gamma 0.1 --wd 1e-4 -e True --resume checkpoints/cifar100/densenet-bc-100-12 --correction conf

  python cifar100.py -a densenet --dataset cifar100 --depth 100 --epochs 500 --schedule 80 120 --gamma 0.1 --wd 1e-4 -e True --resume checkpoints/cifar100/densenet-bc-100-12 --correction focal4
  ```


